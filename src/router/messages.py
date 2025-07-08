import asyncio
import json

from fastapi import APIRouter
from fastapi import Depends, HTTPException, Body
from fastapi.responses import StreamingResponse
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from sqlmodel import Session as DBSession

from src.db.dbs import get_db, add_msg_to_dbs
from src.db.qdrant_client import qdrant_client
from src.db.redis_client import redis_client
from src.models.schema import MsgRequest
from src.router.auth import get_current_user
from src.service.message_service import session_title_gen
from src.service.rag_service import chat_engine
from src.service.set_up_service import get_llm_instance

router = APIRouter()

chat_store = RedisChatStore(
    redis_client=redis_client,
    ttl=3600
)





async def stream_response(engine, session_id, db, title, message):
    full_response = ""

    try:
        yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"

        response = engine.stream_chat(message)
        for token in response.response_gen:
            if token:
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                await asyncio.sleep(0.01)

        yield f"data: {json.dumps({'type': 'done', 'content': full_response})}\n\n"

        if title:
            yield f"data: {json.dumps({'type': 'title', 'content': title})}\n\n"

        handling_save_db(user_msg=message, session_id=session_id, db=db, full_response=full_response)

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


def handling_save_db(session_id, db, full_response, user_msg):
    add_msg_to_dbs(user_msg, session_id, db)
    add_msg_to_dbs(full_response, session_id, db, isUser=False)


# class CurrentState:
#
#     def __init__(self,current_session,current_llm):
#         self.current_session = current_session,
#         self.current_llm = current_llm
#         self.engine = None
#
#
#     def get_current_session(self):
#         return self.current_session
#
#     def set_current_session(self,current_session):
#         self.current_session = current_session
#
#
#     def get_current_llm(self):
#         return self.current_llm
#
#     def set_current_llm(self):
#         self.current_llm = get_llm_instance()
#
#     def set_engine(self,curr_engine):
#         self.engine = curr_engine
#
#     def get_engine(self):
#         return self.engine


def get_memory(session_id):
    memory = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=session_id

    )

    return memory


# curr_engine = SimpleChatEngine.from_defaults(
#     llm=current_model,
#     memory=memory,
#     system_prompt=system_prompt
#
# )


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
        engine_curr = chat_engine.get_engine(session_id=body.session_id, context_id=body.context_id,
                                             context_type=body.context_type)

        title = ""
        if body.isFirst:
            title = await session_title_gen(body.msg)

        return StreamingResponse(
            stream_response(engine_curr, title=title, session_id=body.session_id, db=db, message=body.msg),
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
