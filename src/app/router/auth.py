import datetime
import jwt
from fastapi import Depends, HTTPException,APIRouter
from fastapi.security import OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.app.db.dbs import get_db
from src.app.models.schema import User, RefreshToken


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


router = APIRouter()

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


@router.post("/register", response_model=Token)
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




@router.post("/login", response_model=Token)
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


@router.get("/me")
def me():
    return {"response" : "Hello"}