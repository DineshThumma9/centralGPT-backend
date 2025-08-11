from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict
from uuid import UUID, uuid4

from fastapi import UploadFile
from pydantic import BaseModel, EmailStr
from sqlalchemy import Enum as SQLEnum, Column
from sqlalchemy.dialects.postgresql import JSON
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


class MessageInfo(BaseModel):
    message_id: str
    session_id: str
    content: str
    sender: str
    timestamp: str






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


class MsgRequest(BaseModel):
    session_id: str
    isFirst: bool = False
    msg: str
    context_id:str
    context_type:str
    files:Optional[List[str]]= None



class qdrant_convert(BaseModel):
    point_id: str
    vector: List[float]
    payload: Dict
    collection_name: str


class Notes(BaseModel):
    session_id:str
    context_id:str
    context_type:str



class UserPayload(BaseModel):
    username: str
    email: EmailStr
    password: str


class Token(BaseModel):
    access: str
    refresh: str

class API_KEY_REQUEST(BaseModel):
    api_prov: str
    api_key: str



class TitleUpdateRequest(BaseModel):
    title: str


class TitleResponse(BaseModel):
    title: str


class SessionResponse(BaseModel):
    session_id: str




class git_spec(BaseModel):
    owner:str
    repo:str
    branch:Optional[str] = "main"
    commit:Optional[str] = None
    tree_sha:Optional[str] = None



class GitRequest(BaseModel):
    owner: str
    repo: str
    commit: Optional[str] = None
    branch: Optional[str] = "main"
    dir_include: Optional[List[str]] = None
    dir_exclude: Optional[List[str]] = None
    file_extension_include: Optional[List[str]] = None
    file_extension_exclude: Optional[List[str]] = None
    files:Optional[List[str]] = None


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
    files: Optional[List[str]] = Field(default=None, sa_column=Column(JSON))
    model_response_time_ms: Optional[float] = None


class RefreshToken(SQLModel, table=True):
    __tablename__ = "refreshtoken"
    token: str = Field(primary_key=True)
    email: EmailStr
    expiry_date: datetime

class APIKEYS(SQLModel, table=True):
    __tablename__ = "api_keys"
    user_id: UUID = Field(foreign_key="users.userid", primary_key=True)
    provider: str = Field(primary_key=True, index=True)
    encrypted_key: str


class UserLLMConfig(SQLModel,table=True):
    __tablename__ = "config"
    user_id :UUID = Field(foreign_key="users.userid",primary_key=True)
    provider:str=Field(primary_key=True,index=True)
    model:str



class API_KEY_REQUEST(BaseModel):
    api_prov: str
    api_key: str