import datetime
from uuid import UUID

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel
from sqlmodel import Session as DBSession
from sqlmodel import select

from src.db.dbs import get_db
from src.db.redis_client import redis
from src.models.schema import Message
from src.models.schema import Session as SessionModel
from src.models.schema import User
from src.router.auth import get_current_user

logger.add("logs/api.log", rotation="1 MB", retention="10 days", level="INFO")

logger.info("Server started")

router = APIRouter()

load_dotenv()


class MessageInfo(BaseModel):
    message_id: str
    session_id: str
    content: str
    sender: str
    timestamp: str


class TitleUpdateRequest(BaseModel):
    title: str


class TitleResponse(BaseModel):
    title: str


class SessionResponse(BaseModel):
    session_id: str


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


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str, db: DBSession = Depends(get_db)):
    """Get all messages in a chat session"""
    messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.timestamp).all()
    return messages


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

    session.updated_at = datetime.datetime.utcnow()

    db.add(session)
    db.commit()
    db.refresh(session)

    logger.info(f"Updated title for session {session_id}: {request.title}")
    return TitleResponse(title=session.title)


@router.delete("/{session_id}")
async def delete_session(session_id: str, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
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
    if session_row.user_id != user.userid: raise HTTPException(403)

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
