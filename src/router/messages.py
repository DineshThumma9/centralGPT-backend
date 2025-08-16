import asyncio
import json

from fastapi import APIRouter
from fastapi import Depends, HTTPException, Body
from fastapi.responses import StreamingResponse
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.chat_engine import ContextChatEngine, SimpleChatEngine
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.cohere_rerank import CohereRerank
from loguru import logger
from redisvl.schema import IndexSchema,IndexInfo
from redisvl.utils.rerank import CohereReranker
from sqlalchemy.util import await_only
from sqlmodel import Session as DBSession



from fastapi import Request


from src.db.dbs import get_db, add_msg_to_dbs
from src.db.qdrant_client import  get_vector_store
from src.db.redis_client import chat_store, get_index_store, redis_client
from src.models.schema import MsgRequest
from src.router.auth import get_current_user
from src.service.message_service import session_title_gen, system_prompt
from src.service.rag_service import  code_embed_model, notes_embed_model
from src.service.set_up_service import get_llm_instance

router = APIRouter()


async def streamresponse(engine, session_id, db, title, message,files,request):
    full_response = ""
      # for capturing source info

    try:
        yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"

        # START streaming
        chat_response = engine.stream_chat(message)
        for token in chat_response.response_gen:
            if  await request.is_disconnected():
                yield f"data: {json.dumps({'type': 'abort', 'content': ''})}\n\n"
                logger.info("Stopping Stream")
                break


            if token:
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                await asyncio.sleep(0.01)

        # 🔥 Access source nodes AFTER stream is done
        if hasattr(chat_response, "source_nodes") and chat_response.source_nodes:
            logger.info(f"Sending Source nodes : {chat_response.source_nodes[0]}")
            logger.info(f"Sending Source nodes : {chat_response.source_nodes[0].metadata.keys()}")
            source_nodes = [
                {
                    "score": sn.score,
                    "doc_id": sn.node.id_,
                    "text": sn.node.text[:200],
                    "metadata": sn.node.metadata,

                }
                for sn in chat_response.source_nodes
            ]
            yield f"data: {json.dumps({'type': 'sources', 'content': source_nodes})}\n\n"

        # Send final response
        yield f"data: {json.dumps({'type': 'done', 'content': full_response})}\n\n"
        if title:
            yield f"data: {json.dumps({'type': 'title', 'content': title})}\n\n"

        # Save to DB
        if not await request.is_disconnected():
            handling_save_db(user_msg=message, session_id=session_id, db=db, full_response=full_response,files=files)

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


def handling_save_db(session_id, db, full_response, user_msg,files):
    add_msg_to_dbs(user_msg, session_id, db,files_name=files)
    add_msg_to_dbs(full_response, session_id, db, isUser=False)


def get_memory(session_id):
    memory = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=session_id

    )

    return memory



async def not_ready_stream():
        yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'content': 'Your documents are still being processed. Please wait a moment and try again.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'content': 'Your documents are still being processed. Please wait a moment and try again.'})}\n\n"



@router.post("/simple-stream")
async def message_stream(
        request: Request,
        body: MsgRequest = Body(...),
        db: DBSession = Depends(get_db),
        user=Depends(get_current_user)
):
    current_model = get_llm_instance(db=db, user=user)
    files = body.files
    logger.info(f"Received files in backend:{files}")
    if not current_model:
        raise HTTPException(status_code=401, detail="No model found")

    try:
        memory = get_memory(session_id=body.session_id)
        if body.context_type == "vanilla":
            engine = SimpleChatEngine.from_defaults(
                llm=current_model,
                memory=memory,
                system_prompt=system_prompt
            )
        else:
            collection_name = f"{body.session_id}_{body.context_id}_{body.context_type}"

            # # FIX: Use consistent key pattern
            # status = await redis_client.get(f"collection_name:{collection_name}:status")
            #
            # if not status or status.decode() != "ready":
            #     logger.warning(f"Index not ready for {collection_name}, status: {status}")
            #
            #
            #
            #     return StreamingResponse(
            #         not_ready_stream(),
            #         media_type="text/event-stream",
            #         headers={
            #             "Cache-Control": "no-cache",
            #             "Connection": "keep-alive",
            #             "X-Accel-Buffering": "no",
            #         }
            #     )

            logger.info("Rebuilding ADVANCED RAG pipeline from vector store...")

            # 1. Load the index from the persisted vector store (as before)


            vector_store = get_vector_store(collection_name)

            if vector_store:
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=code_embed_model if body.context_type == 'code' else notes_embed_model
                )

                # 2. Create your custom retriever to fetch more initial candidates (e.g., top 10)
                retriever = index.as_retriever(similarity_top_k=7)

                # 3. Create your reranker to intelligently pick the best results (e.g., top 3)
                reranker = CohereRerank(top_n=3)

                # 4. Build a Query Engine with the retriever and reranker
                # This is the core of your RAG logic

                # 5. Create a chat engine that uses your powerful query engine and memory
                engine = ContextChatEngine.from_defaults(
                    retriever=retriever,
                    node_postprocessors=[reranker],
                    memory=memory,
                    llm=current_model,
                    system_prompt=system_prompt
                )

                logger.info("Advanced RAG pipeline rebuilt successfully.")
            else:
                logger.error("Some how vector stored isnt retivered using simple chat engine temp")

                engine = SimpleChatEngine.from_defaults(
                    llm=current_model,
                    memory=memory,
                    system_prompt=system_prompt
                )




        title = ""
        if body.isFirst:
            title = await session_title_gen(body.msg)

        return StreamingResponse(
            streamresponse(engine, title=title, session_id=body.session_id, db=db, message=body.msg, files=files,
                           request=request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    except Exception as e:
        logger.error(f"Streaming setup error: {e}")
        raise HTTPException(status_code=500, detail="Streaming failed")