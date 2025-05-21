import json
from uuid import uuid4

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlmodel import Session as DBSession

from src.db.dbs import get_db
from src.db.redis_client import redis
from src.models.schema import Message, SenderRole
from src.router.setup import llm_instances

router = APIRouter()


@router.post("/{session_id}")
async def chat(
        session_id: str,
        message: str = Query(...),
        db: DBSession = Depends(get_db)
):
    key = f"chat:{session_id}"

    user_msg = {
        "message_id": str(uuid4()),
        "session_id": session_id,
        "content": message,
        "sender": SenderRole.USER.value
    }
    redis.rpush(key, json.dumps(user_msg))
    db.add(Message(**user_msg))
    db.commit()

    raw_history = redis.lrange(key, -20, -1)
    history = [json.loads(item) for item in raw_history]

    async def event_stream():
        full_response = ""
        async for chunk in llm_instances.ainvoke(history):
            full_response += chunk
            yield chunk

        assistant_msg = {
            "message_id": str(uuid4()),
            "session_id": session_id,
            "content": full_response,
            "sender": SenderRole.ASSISTANT.value
        }
        redis.rpush(key, json.dumps(assistant_msg))
        db.add(Message(**assistant_msg))
        db.commit()

    return StreamingResponse(event_stream(), media_type="text/plain")
