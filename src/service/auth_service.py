import datetime
import os

from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from jose.exceptions import JWTError
from loguru import logger
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from src.db.dbs import get_db
from src.models.schema import User, RefreshToken, APIKEYS, Token

load_dotenv()
logger.info("In Auth")
SECRET_KEY = os.getenv("SECRET_KEY", "YOUR_SECRET_KEY")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRY_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRY_MIN", "360"))
REFRESH_TOKEN_EXPIRY_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRY_DAYS", "30"))
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


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


def get_all_api_keys(user=Depends(get_current_user), db=Depends(get_db)):
    api_val_keys = db.query(APIKEYS).filter(APIKEYS.user_id == user.userid).all()
    return {entry.provider: entry.encrypted_key for entry in api_val_keys}
