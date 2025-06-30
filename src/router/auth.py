import datetime
import os

from dotenv import load_dotenv
from fastapi import Depends, APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError
from loguru import logger
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from src.db.dbs import get_db
from src.models.schema import User, RefreshToken, APIKEYS

load_dotenv()
logger.info("In Auth")
SECRET_KEY = os.getenv("SECRET_KEY", "YOUR_SECRET_KEY")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRY_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRY_MIN", "360"))
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
    now = datetime.datetime.utcnow()

    access_payload = data.copy()
    access_payload["exp"] = now + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRY_MIN)

    refresh_payload = data.copy()
    refresh_payload["exp"] = now + datetime.timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS)

    try:
        access_token = jwt.encode(access_payload, SECRET_KEY, algorithm=ALGORITHM)
        refresh_token = jwt.encode(refresh_payload, SECRET_KEY, algorithm=ALGORITHM)
    except Exception as e:
        logger.exception("JWT encoding failed")
        return JSONResponse(content={"detail": "Token generation failed"}, status_code=500)

    email = data.get("sub")
    db_token = RefreshToken(
        email=email,
        token=refresh_token,
        expiry_date=now + datetime.timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS)
    )

    try:
        db.add(db_token)
        db.commit()
        logger.info(f"Refresh token stored for user: {email}")
    except Exception as e:
        db.rollback()
        logger.exception("Failed to store refresh token in DB")
        return JSONResponse(content={"detail": "Database error"}, status_code=500)

    tokens = Token(access=access_token, refresh=refresh_token)
    return JSONResponse(content=tokens.model_dump(), status_code=200)

@router.post("/register", response_model=Token)
def register(user: UserPayload, db: Session = Depends(get_db)):
    logger.info("Accessing register")
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
        logger.exception("User registration failed")
        return JSONResponse(content={"detail": f"Registration failed: {str(e)}"}, status_code=500)

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not pwd_context.verify(form_data.password, user.hpassword):
        raise HTTPException(status_code=401, detail="Invalid username or password")

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
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")



async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
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
    except JWTError:
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


def get_all_api_keys(user=Depends(get_current_user),db=Depends(get_db)):
    api_val_keys =  db.query(APIKEYS).filter(APIKEYS.user_id == user.userid).all()
    return {entry.provider:entry.encrypted_key for entry in api_val_keys}

