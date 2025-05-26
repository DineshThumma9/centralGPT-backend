import datetime
import json
import logging
import os
import jwt


from dotenv import load_dotenv
from fastapi import Depends, APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from src.db.dbs import get_db
from src.models.schema import User, RefreshToken

from loguru import  logger

load_dotenv()
logger.info("In Auth")
SECRET_KEY = os.getenv("SECRET_KEY", "YOUR_SECRET_KEY")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRY_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRY_MIN", "15"))
REFRESH_TOKEN_EXPIRY_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRY_DAYS", "30"))




pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
router = APIRouter()






class UserPayload(BaseModel):
    username: str
    email: EmailStr
    password: str


class Token(BaseModel):
    access: str
    refresh: str





def create_tokens(data: dict, db: Session):
    now = datetime.datetime.now()
    access_payload = data.copy()
    access_payload["exp"] = now + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRY_MIN)
    refresh_payload = data.copy()
    refresh_payload["exp"] = now + datetime.timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS)

    access_token = jwt.encode(access_payload, SECRET_KEY, algorithm=ALGORITHM)
    refresh_token = jwt.encode(refresh_payload, SECRET_KEY, algorithm=ALGORITHM)

    email = data.get("sub")
    db_token = RefreshToken(
        email=email,
        token=refresh_token,
        expiry_date=now + datetime.timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS)
    )
    db.add(db_token)
    db.commit()


    tokens = Token(access=access_token, refresh=refresh_token)
    return JSONResponse(
            content=tokens.model_dump(),
            status_code=200
        )








@router.post("/register", response_model=Token)
def register(user: UserPayload, db: Session = Depends(get_db)):



    logger.info("Accessiong register")
    logger.info(user)
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        return JSONResponse(content={"detail": "Email already registered"}, status_code=400)

    hashed_pw = pwd_context.hash(user.password)
    new_user = User(username=user.username, email=user.email, hpassword=hashed_pw)

    try:
        db.add(new_user)
        db.commit()
        return create_tokens({"sub": user.email}, db)

    except Exception as e:
        db.rollback()
        return JSONResponse(content={"detail": f"Registration failed: {str(e)}"}, status_code=500)









@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not pwd_context.verify(form_data.password, user.hpassword):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return create_tokens({"sub": user.email}, db)





@router.post("/refresh", response_model=Token)
def refresh_token(refresh: str = Form(...), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(refresh, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid refresh token")

        db_token = db.query(RefreshToken).filter(RefreshToken.token == refresh).first()
        if not db_token:
            raise HTTPException(status_code=401, detail="Refresh token not found")

        return create_tokens({"sub": email}, db)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")






async def get_current_user(token:str = Depends(OAuth2PasswordBearer(tokenUrl="token")), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user





@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "user_id": str(current_user.userid),
        "username": current_user.username,
        "email": current_user.email
    }
