from fastapi import Depends, APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError
from loguru import logger
from sqlalchemy.orm import Session

from src.db.dbs import get_db
from src.models.schema import User, RefreshToken, UserPayload, Token
from src.service.auth_service import pwd_context, create_tokens, SECRET_KEY, ALGORITHM, \
    get_current_user

router = APIRouter()


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


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "user_id": str(current_user.userid),
        "username": current_user.username,
        "email": current_user.email
    }
