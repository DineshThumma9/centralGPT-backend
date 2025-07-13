import asyncio
import json

from fastapi import APIRouter
from fastapi import Depends, HTTPException, Body
from fastapi.responses import StreamingResponse
from llama_index.core.chat_engine import ContextChatEngine, SimpleChatEngine
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from sqlmodel import Session as DBSession

from src.db.dbs import get_db, add_msg_to_dbs
from src.db.qdrant_client import qdrant_client
from src.db.redis_client import redis_client
from src.models.schema import MsgRequest
from src.router.auth import get_current_user
from src.service.message_service import session_title_gen, system_prompt
from src.service.rag_service import chat_engine
from src.service.set_up_service import get_llm_instance

router = APIRouter()

chat_store = RedisChatStore(
    redis_client=redis_client,
    ttl=3600
)


async def stream_response(engine, session_id, db, title, message):
    full_response = ""
    source_nodes = None  # for capturing source info

    try:
        yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"

        # START streaming
        chat_response = engine.stream_chat(message)

        for token in chat_response.response_gen:
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
        handling_save_db(user_msg=message, session_id=session_id, db=db, full_response=full_response)

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"



def handling_save_db(session_id, db, full_response, user_msg):
    add_msg_to_dbs(user_msg, session_id, db)
    add_msg_to_dbs(full_response, session_id, db, isUser=False)



def get_memory(session_id):
    memory = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=session_id

    )

    return memory




vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="llamaIndex"
)


@router.post("/simple-stream")
async def message_stream(
        body: MsgRequest = Body(...),
        db: DBSession = Depends(get_db),
        user=Depends(get_current_user)
):
    current_model = get_llm_instance(db=db, user=user)
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
            ret = chat_engine.get_engine(session_id=body.session_id, context_id=body.context_id,
                                         context_type=body.context_type)

            engine = ContextChatEngine.from_defaults(
                retriever=ret,
                llm=current_model,
                system_prompt=system_prompt,
                vector_store=vector_store,
                memory=memory,

            )

        title = ""
        if body.isFirst:
            title = await session_title_gen(body.msg)

        return StreamingResponse(
            stream_response(engine, title=title, session_id=body.session_id, db=db, message=body.msg),
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
