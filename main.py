


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import Query
from fastapi.responses import JSONResponse, StreamingResponse
from src.app.application import app
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import datetime
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Literal
import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr
from sqlmodel import SQLModel, Field
from sqlalchemy.ext.declarative import declarative_base

from src.models import RefreshToken, User

DATABASE_URL="postgresql://postgres:yourpassword@localhost:5432/postgres"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autobind = False, autocommit = False,bind = engine)
Base = declarative_base()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5176"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.state.llm_class = None
app.state.llm_instance = None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_table():
    Base.metadata.create_all(bind=engine)

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
        created_at: datetime = Field(default_factory=datetime.now())

    class Session(SQLModel, table=True):
        __tablename__ = "sessions"
        session_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
        user_id: uuid.UUID = Field(foreign_key="users.userid", index=True)
        title: str
        model: str
        created_at: datetime = Field(default_factory=datetime.now())
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

    class RefreshToken(SQLModel, table=True):
        __tablename__ = "refreshtoken"
        token: str = Field(primary_key=True)
        email: EmailStr
        expiry_date: datetime




pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



SECRET_KEY = "YOUR_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRY_MIN = 15
REFRESH_TOKEN_EXPIRY_DAYS = 30


class UserPayload(BaseModel):
    username: str
    email: str
    password: str


class Token(BaseModel):
    access: str
    refresh: str
    type: str = "bearer"



def create_tokens(data: dict):
    now = datetime.datetime.utcnow()

    access_payload = data.copy()
    access_payload["exp"] = now + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRY_MIN)

    refresh_payload = data.copy()
    refresh_payload["exp"] = now + datetime.timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS)

    access_token = jwt.encode(access_payload, SECRET_KEY, algorithm=ALGORITHM)
    refresh_token = jwt.encode(refresh_payload, SECRET_KEY, algorithm=ALGORITHM)

    return access_token, refresh_token


@app.post("/register", response_model=Token)
async def register(user: UserPayload, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = pwd_context.hash(user.password)
    new_user = User(username=user.username, email=user.email, hpassword=hashed_pw)

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    access_token, refresh_token = create_tokens({"sub": user.email})

    db_token = RefreshToken(
        email=user.email,
        token=refresh_token,
        expiry_date=datetime.datetime.utcnow() + datetime.timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS)
    )

    db.add(db_token)
    db.commit()

    return Token(access=access_token, refresh=refresh_token)




@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()  # OAuth2 uses `username` field for email

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not pwd_context.verify(form_data.password, user.hpassword):
        raise HTTPException(status_code=401, detail="Incorrect password")

    access_token, refresh_token = create_tokens({"sub": user.email})

    db_token = RefreshToken(
        email=user.email,
        token=refresh_token,
        expiry_date=datetime.datetime.utcnow() + datetime.timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS)
    )

    db.add(db_token)
    db.commit()

    return Token(access=access_token, refresh=refresh_token)


@app.get("/me")
def me():
    return {"response" : "Hello"}


llms = {
    "groq": ChatGroq,
    "ollama": ChatOllama,
}

@app.post("/api/{api_key}")
def get_api_key(api_key: str):
    os.environ["GROQ_API_KEY"] = api_key

@app.get("/models/{llm_prov}")
def choose_llm_provider(llm_prov: str):
    llm_class = llms.get(llm_prov)
    if not llm_class:
        raise HTTPException(status_code=404, detail=f"LLM provider '{llm_prov}' not found")

    app.state.llm_class = llm_class
    return JSONResponse(
        content={"message": f"LLM provider '{llm_prov}' selected successfully"},
        status_code=200
    )

@app.get("/models/model/{model}")
def choose_model(model: str):
    llm_class = getattr(app.state, "llm_class", None)
    if llm_class is None:
        raise HTTPException(status_code=400, detail="No LLM provider selected")

    try:
        llm_instance = llm_class(model=model)
        app.state.llm_instance = llm_instance
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to instantiate model: {str(e)}")

    return JSONResponse(
        content={"message": f"Model '{model}' instantiated successfully"},
        status_code=200
    )

@app.api_route("/chat", methods=["GET", "POST"])
def getResponse(message: str = Query(...)):
    if not hasattr(app.state, "llm_instance"):
        raise HTTPException(status_code=400, detail="No model instance selected")

    response = app.state.llm_instance.invoke(message)

    if hasattr(response, "content"):
        return JSONResponse(
            content={"response": response.content},
            status_code=200
        )
    else:
        return JSONResponse(
            content={"response": str(response)},
            status_code=200
        )

@app.post("/chat/stream")
def streamMsg(query: str):
    def event_stream():
        for chunk in app.state.llm_instance.astream(query):
            yield chunk
    return StreamingResponse(event_stream(), media_type="text/plain")