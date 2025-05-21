from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr
from sqlalchemy import Enum as SQLEnum
from sqlmodel import SQLModel, Field


# --- Shared Types ---
class SenderRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# --- API (Pydantic) ---
class ChatMessage(BaseModel):
    role: SenderRole
    content: str
    timestamp: Optional[datetime] = None


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, str]


class ChatbotSchema(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False
    extra: Optional[Dict] = None


class ModelInfo(BaseModel):
    modelprovider: str
    modelname: str
    isFunctionCalling: bool
    token_left: float


# --- DB (SQLModel) ---
class User(SQLModel, table=True):
    __tablename__ = "users"
    userid: UUID = Field(default_factory=uuid4, primary_key=True)
    username: str = Field(index=True)
    email: EmailStr = Field(index=True)
    hpassword: str
    created_at: datetime = Field(default_factory=datetime.now)


class Session(SQLModel, table=True):
    __tablename__ = "sessions"
    session_id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="users.userid", index=True)
    title: str
    model: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Message(SQLModel, table=True):
    __tablename__ = "messages"
    message_id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.session_id", index=True)
    sender: SenderRole = Field(sa_column=SQLEnum(SenderRole))
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    model_response_time_ms: Optional[float] = None


class RefreshToken(SQLModel, table=True):
    __tablename__ = "refreshtoken"
    token: str = Field(primary_key=True)
    email: EmailStr
    expiry_date: datetime
