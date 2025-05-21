import datetime
import jwt
import os
from fastapi import Depends, APIRouter, HTTPException
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from src.db.dbs import get_db
from src.models.schema import User, RefreshToken

# Load environment variables
load_dotenv()

# Authentication settings from environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "YOUR_SECRET_KEY")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRY_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRY_MIN", "15"))
REFRESH_TOKEN_EXPIRY_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRY_DAYS", "30"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
router = APIRouter()

from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session


class OAuth2PasswordRequestFormEmail(OAuth2PasswordRequestForm):
    def __init__(self, grant_type: str = None, email: str = None, password: str = None, scope: str = "", client_id: str = None, client_secret: str = None):
        super().__init__(grant_type, email, password, scope, client_id, client_secret)
        self.email = email


@router.post("/login")
async def login(form_data: OAuth2PasswordRequestFormEmail = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.email).first()
    if not user or not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(status_code=422, detail="Invalid email or password")

    if not pwd_context.verify(form_data.password, user.hpassword):
        return JSONResponse(content={"detail": "Invalid credentials"}, status_code=401)

    # Create tokens
    access_token, refresh_token = create_tokens({"sub": user.email})

    # Store refresh token
    db_token = RefreshToken(
        email=user.email,
        token=refresh_token,
        expiry_date=datetime.datetime.utcnow() + datetime.timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS)
    )
    db.add(db_token)
    db.commit()

    return Token(access=access_token, refresh=refresh_token)

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
    except jwt.PyJWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user


@router.post("/register", response_model=Token)
def register(user: UserPayload, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        return JSONResponse(
            content={"detail": "Email already registered"},
            status_code=400
        )

    # Create new user
    hashed_pw = pwd_context.hash(user.password)
    new_user = User(username=user.username, email=user.email, hpassword=hashed_pw)

    try:
        db.add(new_user)
        db.commit()

        # Create tokens
        access_token, refresh_token = create_tokens({"sub": user.email})

        # Store refresh token
        db_token = RefreshToken(
            email=user.email,
            token=refresh_token,
            expiry_date=datetime.datetime.utcnow() + datetime.timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS)
        )
        db.add(db_token)
        db.commit()

        return Token(access=access_token, refresh=refresh_token)
    except Exception as e:
        db.rollback()
        return JSONResponse(
            content={"detail": f"Registration failed: {str(e)}"},
            status_code=500
        )





@router.post("/refresh", response_model=Token)
def refresh_token(refresh: str, db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(refresh, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid refresh token")

        # Verify refresh token exists in database
        db_token = db.query(RefreshToken).filter(RefreshToken.token == refresh).first()
        if not db_token:
            raise HTTPException(status_code=401, detail="Refresh token not found")

        # Create new tokens
        access_token, new_refresh_token = create_tokens({"sub": email})

        # Update token in database
        db_token.token = new_refresh_token
        db_token.expiry_date = datetime.datetime.utcnow() + datetime.timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS)
        db.commit()

        return Token(access=access_token, refresh=new_refresh_token)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "user_id": str(current_user.userid),
        "username": current_user.username,
        "email": current_user.email
    }