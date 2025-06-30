import json
import logging
import os
from datetime import datetime
from typing import Generator
from uuid import UUID, uuid4

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from src.db.redis_client import redis
from src.models.schema import Message, SenderRole

logger = logging.getLogger("database")
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/postgres")

try:
    logger.info(f"Creating database connection to {DATABASE_URL.split('@')[-1]}")
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={
            "connect_timeout": 5
        }
    )

    SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)
    logger.info("Database connection established")

except Exception as e:
    logger.error(f"Failed to connect to database: {str(e)}")
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        logger.debug("Database session created")
        yield db
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        logger.debug("Database session closed")
        db.close()


def create_all_tables():
    try:
        logger.info("Creating database tables")
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        raise


from pydantic import BaseModel


class MessageInfo(BaseModel):
    message_id: str
    session_id: str
    content: str
    sender: str
    timestamp: str


def add_msg_to_dbs(msg: str, session_id: str, db: Session, isUser: bool = True):
    message = Message(
        message_id=uuid4(),
        session_id=UUID(session_id),
        content=msg,
        sender=SenderRole.USER if isUser else SenderRole.ASSISTANT,
        timestamp=datetime.now()
    )

    redis_key_prefix = f"chat_session:{session_id}"

    db.add(message)
    db.commit()
    db.refresh(message)

    json_msg = json.dumps(message.model_dump(mode="json"))

    redis.rpush(f"{redis_key_prefix}:messages", json_msg)

    if not isUser:
        return message.model_dump(mode="json")
