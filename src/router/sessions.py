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
from sqlmodel import select
from sqlmodel import Session as DBSession

from src.db.dbs import get_db
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


class SessionResponse(BaseModel):
    session_id: str
    message: str


class SessionInfo(BaseModel):
    session_id: str
    title: str
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
            session_id=str(new_session.session_id),
            message="Session created successfully"
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


@router.get("/history/{session_id}")
async def get_chat_history(
        session_id: str,
        limit: Optional[int] = Query(50, ge=1, le=100),
        db: DBSession = Depends(get_db)
):
    """Get chat history for a session"""
    try:

        session_query = select(SessionModel).where(SessionModel.session_id == UUID(session_id))
        session = db.execute(session_query).scalars().all()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        message_query = select(Message).where(
            Message.session_id == UUID(session_id)
        ).order_by(Message.timestamp).limit(limit)

        messages = db.execute(message_query).scalars().all()

        # Format messages to match frontend expectations
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "message_id": str(msg.message_id),
                "session_id": str(msg.session_id),
                "content": msg.content,
                "role": msg.sender.value,
                "created_at": msg.timestamp.isoformat()
            })

        return formatted_messages

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")
    except Exception as e:
        logger.error(f"Error fetching history for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch chat history")



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


# Get all sessions for a user
@router.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str, db: DBSession = Depends(get_db)):
    """Get all chat sessions for a user"""
    sessions = db.query(Message.session_id).filter(
        Message.session_id.like(f"{user_id}_%")
    ).distinct().all()

    session_list = []
    for session in sessions:
        session_id = session[0]
        # Get first message for preview
        first_msg = db.query(Message).filter(
            Message.session_id == session_id,
            Message.sender == SenderRole.USER
        ).order_by(Message.timestamp).first()

        # Get last activity
        last_msg = db.query(Message).filter(
            Message.session_id == session_id
        ).order_by(Message.timestamp.desc()).first()

        session_list.append({
            "session_id": session_id,
            "preview": first_msg.content[:50] + "..." if first_msg else "New Chat",
            "last_activity": last_msg.timestamp if last_msg else None
        })

    return sorted(session_list, key=lambda x: x["last_activity"] or datetime.min, reverse=True)


# Get chat history for a specific session
@router.get("/chat/{session_id}")
async def get_chat_history(session_id: str, db: DBSession = Depends(get_db)):
    """Get all messages in a chat session"""
    messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.timestamp).all()
    return messages


# Create new session
@router.post("/session/new")
async def create_new_session(
        user_id: str = Body(...),
        db: DBSession = Depends(get_db)
):
    """Create a new chat session"""
    session_id = f"{user_id}_{uuid4()}"

    # Optional: Store in Redis for quick access
    redis.sadd(f"user_sessions:{user_id}", session_id)

    return {"session_id": session_id}


@router.post("/simple/{msg}")
async def message(
        msg: str,
        request: Request,
        db: DBSession = Depends(get_db),
        body: dict = Body(...)
):
    current_model = getattr(request.app.state, "current_model", None)
    if not (current_model and hasattr(current_model, "invoke")):
        raise HTTPException(status_code=401, detail="No valid model found")

    session_id = body.get("current_session_id")
    if not session_id:
        raise HTTPException(status_code=422, detail="Missing current_session_id in request body")

    # ✅ CRITICAL: Ensure unique Redis keys
    redis_key_prefix = f"chat_session:{session_id}"

    try:
        # --- Setup isolated memory with unique Redis keys ---
        chat_history = RedisChatMessageHistory(
            session_id=redis_key_prefix,  # Use prefixed key
            url="redis://localhost:6379",
            key_prefix="langchain:chat_history:",  # Additional prefix
            ttl=3600  # Optional: expire after 1 hour
        )

        # ✅ Create fresh memory instance for each request
        memory = ConversationBufferMemory(
            chat_memory=chat_history,
            return_messages=True,
            memory_key="history"  # Explicit memory key
        )

        # --- Log current session for debugging ---
        logger.info(f"Processing message for session: {session_id}")
        logger.info(f"Redis key: {redis_key_prefix}")

        # --- Store user message ---
        user_msg = Message(
            session_id=session_id,
            message_id=uuid4(),
            content=msg,
            sender=SenderRole.USER,
            timestamp=datetime.now()
        )
        db.add(user_msg)
        db.commit()
        db.refresh(user_msg)

        # Store in Redis with unique key
        redis.rpush(f"{redis_key_prefix}:messages", json.dumps(user_msg.model_dump()))

        # ✅ Create fresh conversation chain for each request
        chain = ConversationChain(
            llm=current_model,
            memory=memory,
            verbose=True  # Enable for debugging
        )

        # ✅ Clear any potential model state (Groq specific)
        # Force fresh context by explicitly providing input
        response = chain.invoke({
            "input": msg,
            "history": memory.chat_memory.messages  # Explicit history
        })

        response_text = response.get('response', str(response)) if isinstance(response, dict) else str(response)
        logger.info(f"Generated response for session {session_id}: {response_text[:100]}...")

        # --- Store assistant response ---
        assistant_msg = Message(
            session_id=session_id,
            message_id=uuid4(),
            content=response_text,
            sender=SenderRole.ASSISTANT,
            timestamp=datetime.now()
        )
        db.add(assistant_msg)
        db.commit()
        db.refresh(assistant_msg)

        # Store in Redis
        assistant_msg_info = MessageInfo(
            message_id=str(assistant_msg.message_id),
            session_id=session_id,
            content=response_text,
            sender=SenderRole.ASSISTANT,
            timestamp=str(assistant_msg.timestamp)
        )
        redis.rpush(f"{redis_key_prefix}:messages", json.dumps(assistant_msg_info.model_dump()))

        # --- Optional: Clear memory after response (prevents bleeding) ---
        # Uncomment if you want to clear memory after each response
        # memory.clear()

        return assistant_msg_info

    except Exception as e:
        logger.error(f"Error in message processing for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ✅ Add endpoint to clear session memory (for testing)
@router.delete("/session/{session_id}/clear")
async def clear_session_memory(session_id: str):
    """Clear memory for a specific session"""
    redis_key_prefix = f"chat_session:{session_id}"

    # Clear Redis chat history
    redis.delete(f"langchain:chat_history:{redis_key_prefix}")
    redis.delete(f"{redis_key_prefix}:messages")

    return {"message": f"Cleared memory for session {session_id}"}


# ✅ Add endpoint to check session state (for debugging)
@router.get("/session/{session_id}/debug")
async def debug_session(session_id: str):
    """Debug session state"""
    redis_key_prefix = f"chat_session:{session_id}"

    # Check Redis keys
    chat_history_key = f"langchain:chat_history:{redis_key_prefix}"
    messages_key = f"{redis_key_prefix}:messages"

    chat_history = redis.lrange(chat_history_key, 0, -1)
    messages = redis.lrange(messages_key, 0, -1)

    return {
        "session_id": session_id,
        "redis_prefix": redis_key_prefix,
        "chat_history_count": len(chat_history),
        "messages_count": len(messages),
        "chat_history": [json.loads(msg) if msg else None for msg in chat_history[:5]],  # First 5
        "messages": [json.loads(msg) if msg else None for msg in messages[:5]]  # First 5
    }
# Alternative: Get messages from Redis (faster)
@router.get("/chat/{session_id}/redis")
async def get_chat_from_redis(session_id: str):
    """Get chat history from Redis (faster but less reliable)"""
    messages = redis.lrange(f"{session_id}:messages", 0, -1)
    return [json.loads(msg) for msg in messages]


# Hybrid approach: Redis first, fallback to DB
@router.get("/chat/{session_id}/hybrid")
async def get_chat_hybrid(session_id: str, db: DBSession = Depends(get_db)):
    """Get chat history - try Redis first, fallback to database"""
    try:
        # Try Redis first (faster)
        redis_messages = redis.lrange(f"{session_id}:messages", 0, -1)
        if redis_messages:
            return [json.loads(msg) for msg in redis_messages]
    except Exception as e:
        logger.warning(f"Redis error, falling back to DB: {e}")

    # Fallback to database
    messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.timestamp).all()

    # Populate Redis for next time
    try:
        for msg in messages:
            msg_info = MessageInfo(
                message_id=str(msg.message_id),
                session_id=msg.session_id,
                content=msg.content,
                sender=msg.sender,
                timestamp=str(msg.timestamp)
            )
            redis.rpush(f"{session_id}:messages", json.dumps(msg_info.model_dump()))
    except Exception as e:
        logger.warning(f"Failed to populate Redis: {e}")

    return messages

# Assume these are imported:
# SessionModel, Message, TitleUpdateRequest, TitleResponse, get_db, redis

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
    #    if session_row.user_id != current_user.id: raise HTTPException(403)

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

    return {"message": "Session deleted successfully"}


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


