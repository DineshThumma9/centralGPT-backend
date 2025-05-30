import datetime
import json
import os
from datetime import datetime
from typing import Optional, List, Dict
from urllib import request
from uuid import UUID
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException,Body
from fastapi import Query, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.sql.functions import current_user
from sqlmodel import select
from sqlmodel import Session as DBSession

from src.db.dbs import get_db, add_msg_to_dbs
from src.db.redis_client import redis
from src.models.schema import Message, SenderRole
from src.models.schema import Session as SessionModel
from src.models.schema import User
from src.router.auth import get_current_user
from src.router.setup import llm_instances

logger.add("logs/api.log", rotation="1 MB", retention="10 days", level="INFO")

logger.info("Server started")

router = APIRouter()

load_dotenv()



class SessionInfo(BaseModel):
    session_id: str
    user_id : str
    model:Optional[str]
    title: str = "New Chat"
    created_at: str
    updated_at: Optional[str] = None


class MessageInfo(BaseModel):
    message_id: str
    session_id: str
    content: str
    sender: str
    timestamp:str



class TitleUpdateRequest(BaseModel):
    title: str


class TitleResponse(BaseModel):
    title: str


class SessionResponse(BaseModel):
    session_id : str


@router.post("/new", response_model=SessionResponse)
async def create_new_session(user: User = Depends(get_current_user), db: DBSession = Depends(get_db)):
    try:

        try:
            new_session = SessionModel(
                user_id=user.userid,
                title="New Chat",
                model="default",
            )
        except Exception as model_error:
            raise HTTPException(status_code=422, detail=f"Model creation failed: {str(model_error)}")

        db.add(new_session)
        db.commit()
        db.refresh(new_session)

        return SessionResponse(
            session_id=str(new_session.session_id)
        )

    except HTTPException:
        raise
    except Exception as e:
        try:
            db.rollback()
        except:
            pass

        raise HTTPException(
            status_code=500,
            detail=f"Session creation failed: {str(e)}"
        )



class qdrant_convert(BaseModel):
    point_id:str
    vector:List[float]
    payload:Dict
    collection_name:str


from sentence_transformers import SentenceTransformer



def conversion_for_qdrant(msg: MessageInfo, collection_name:str):
    msg_id = msg.message_id  # str, not tuple

    embed_model =  SentenceTransformer("all-MiniLM-L6-v2")
    vector = embed_model.encode(msg.content)
    payload = {
        "content": msg.content,
        "sender": msg.sender,
        "timestamp": msg.timestamp
    }

    return qdrant_convert(point_id=msg_id, vector=vector, payload=payload,collection_name=collection_name)



from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory.chat_message_histories.redis import RedisChatMessageHistory
from langchain.schema import HumanMessage
from fastapi import APIRouter, Request, Body, Depends, HTTPException
from uuid import uuid4
from datetime import datetime


@router.get("/chat/{session_id}")
async def get_chat_history(session_id: str, db: DBSession = Depends(get_db)):
    """Get all messages in a chat session"""
    messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.timestamp).all()
    return messages




class MsgRequest(BaseModel):
    session_id : str
    msg : str

@router.post("/simple")
async def message(
    request: Request,
    db: DBSession = Depends(get_db),
    body: MsgRequest = Body(...)
):




    current_model = getattr(request.app.state, "current_model", None)
    if not (current_model and hasattr(current_model, "invoke")):
        raise HTTPException(status_code=401, detail="No valid model found")

    session_id = body.session_id
    if not session_id:
        raise HTTPException(status_code=422, detail="Missing current_session_id in request body")

    redis_key_prefix = f"chat_session:{session_id}"



    try:
        chat_history = RedisChatMessageHistory(
            session_id=redis_key_prefix,
            url="redis://localhost:6379",
            key_prefix="langchain:chat_history:",
            ttl=3600
        )

        memory = ConversationBufferMemory(
            chat_memory=chat_history,
            return_messages=True,
            memory_key="history"
        )

        from src.db.dbs import add_msg_to_dbs

        add_msg_to_dbs(body.msg,session_id,db)


        chain = ConversationChain(
            llm=current_model,
            memory=memory,
            verbose=True
        )

        response = chain.invoke({
            "input": body.msg,
            "history": memory.chat_memory.messages
        })

        response_text = response.get('response', str(response)) if isinstance(response, dict) else str(response)



        assistant_msg_info = add_msg_to_dbs(response_text,session_id,db,False)

        return assistant_msg_info

    except Exception as e:
        logger.error(f"Error in message processing for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")




@router.patch("/{session_id}/title", response_model=TitleResponse)
async def update_session_title(
        session_id: str,
        request: TitleUpdateRequest,
        db: DBSession = Depends(get_db)
):
    """Update session title"""
    try:
        session_uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    session_query = select(SessionModel).where(SessionModel.session_id == session_uuid)
    session = db.execute(session_query).scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.title = request.title


    session.updated_at = datetime.utcnow()

    db.add(session)
    db.commit()
    db.refresh(session)

    logger.info(f"Updated title for session {session_id}: {request.title}")
    return TitleResponse(title=session.title)



@router.delete("/{session_id}")
async def delete_session(session_id: str, db: DBSession = Depends(get_db)):
    # 1. Parse & validate UUID
    try:
        sid = UUID(session_id)
    except ValueError:
        raise HTTPException(400, "Invalid session ID format")

    # 2. Fetch the Session row
    stmt = select(SessionModel).where(SessionModel.session_id == sid)
    result = db.execute(stmt)
    session_row = result.scalars().first()
    if not session_row:
        raise HTTPException(404, "Session not found")

    # 3. (Optional) Authorization check here:
    if session_row.user_id != current_user.id: raise HTTPException(403)

    # 4. Delete all Messages linked to it
    msg_stmt = select(Message).where(Message.session_id == sid)
    messages = db.execute(msg_stmt).scalars().all()
    for m in messages:
        db.delete(m)

    # 5. Delete the Session row itself
    db.delete(session_row)
    db.commit()

    # 6. Redis cleanup, etc.
    try:
        redis.delete(f"chat:{session_id}")
    except:
        pass

    return True


@router.get("/getAll")
async def get_all_sessions(
    db: DBSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Get all sessions for the current user"""
    try:
        session_query = (
            select(SessionModel)
            .where(SessionModel.user_id == user.userid)
            .order_by(SessionModel.updated_at)
        )
        sessions = db.execute(session_query).scalars().all()

        return [
            {
                "id": str(session.session_id),
                "session_id": str(session.session_id),
                "title": session.title,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat() if session.updated_at else None
            }
            for session in sessions
        ]

    except Exception as e:
        logger.error(f"Error fetching sessions for user {user.userid}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch sessions")


@router.get("/message/stream")
async def stream_chat_response(
        sessionId: str = Query(...),
        message: str = Query(...),
        db: DBSession = Depends(get_db)
):
    """Stream chat response using Server-Sent Events"""
    try:
        # Validate session exists
        session_query = select(SessionModel).where(SessionModel.session_id == UUID(sessionId))
        session = db.execute(session_query).first()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Save user message
        user_msg_id = str(uuid4())
        user_message = Message(
            message_id=UUID(user_msg_id),
            session_id=UUID(sessionId),
            sender=SenderRole.USER,
            content=message,
            timestamp=datetime.now()
        )

        db.add(user_message)
        db.commit()

        # Cache in Redis
        redis_key = f"chat:{sessionId}"
        user_msg_data = {
            "message_id": user_msg_id,
            "session_id": sessionId,
            "content": message,
            "sender": SenderRole.USER.value,
            "timestamp": user_message.timestamp.isoformat()
        }
        redis.rpush(redis_key, json.dumps(user_msg_data))

        # Get conversation history
        raw_history = redis.lrange(redis_key, -20, -1)
        history = [json.loads(item) for item in raw_history]

        async def generate_sse():
            try:
                assistant_msg_id = str(uuid4())
                full_response = ""

                # Send initial message with assistant message ID
                yield f"data: {json.dumps({'type': 'start', 'message_id': assistant_msg_id})}\n\n"

                # Stream LLM response
                async for chunk in llm_instances.ainvoke(history):
                    full_response += chunk
                    yield f"data: {json.dumps({'type': 'content', 'message_id': assistant_msg_id, 'content': chunk})}\n\n"

                # Save complete assistant message
                assistant_message = Message(
                    message_id=UUID(assistant_msg_id),
                    session_id=UUID(sessionId),
                    sender=SenderRole.ASSISTANT,
                    content=full_response,
                    timestamp=datetime.now()
                )

                db.add(assistant_message)




                db.commit()

                # Cache in Redis
                assistant_msg_data = {
                    "message_id": assistant_msg_id,
                    "session_id": sessionId,
                    "content": full_response,
                    "sender": SenderRole.ASSISTANT.value,
                    "timestamp": assistant_message.timestamp.isoformat()
                }
                redis.rpush(redis_key, json.dumps(assistant_msg_data))

                # Send completion signal
                yield f"data: {json.dumps({'type': 'complete', 'message_id': assistant_msg_id})}\n\n"

            except Exception as e:
                logger.error(f"Error in stream generation: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")
    except Exception as e:
        logger.error(f"Error in stream chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process chat message")

