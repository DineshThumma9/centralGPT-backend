from typing import Optional, List, Dict, Literal, Any
import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr
from sqlmodel import SQLModel, Field, JSON




class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatbotSchema(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float
    max_tokens: int
    stream: bool
    extra: Optional[Dict]


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, str]



class ModelInfo(BaseModel):
    modelprovider: str
    modelname: str
    isFunctionCalling: bool
    token_left: float


class User(SQLModel, table=True):
    __tablename__ = "users"
    userid: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    username: str = Field(index=True)
    email: EmailStr = Field(index=True)
    hpassword: str
    created_at: datetime = Field(default_factory=datetime.now)

class Session(SQLModel, table=True):
    __tablename__ = "sessions"
    session_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="users.userid", index=True)
    title: str
    model: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

from enum import Enum
from sqlalchemy import Enum as SQLEnum

class SenderType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class Message(SQLModel, table=True):
    __tablename__ = "messages"
    message_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    session_id: uuid.UUID = Field(foreign_key="sessions.session_id", index=True)
    sender: SenderType = Field(sa_column=SQLEnum(SenderType))
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_response_time_ms: Optional[float] = None


class RefreshToken(SQLModel,table=True):
    __tablename__ = "refreshtoken"
    token:str=Field(primary_key=True)
    email:EmailStr
    expiry_date:datetime
