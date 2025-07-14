import contextlib
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.requests import Request

from src.db.dbs import create_all_tables
from src.router import auth_router, basic_router, session_router,message_router,rag_router

app = FastAPI(
    title="FastAPI App",
    description="Simple FastAPI Application",
    version="1.0.0"
)

logger.add("logs/api.log", rotation="1 MB", retention="10 days", level="INFO")
logger.info("Server started")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:55000",
        "http://localhost:55000",
        "http://localhost:55001",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "https://central-gpt-frontend.vercel.app",
        "https://central-gpt-frontend-4svito8yj-ducts-projects.vercel.app",
    ],
    allow_credentials=True,
    allow_origin_regex=r"^https:\/\/central-gpt\.vercel\.app$",
    allow_methods=["*"],
    allow_headers=["*"],
)


import time
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration}ms)")
    return response



@app.get("/")
@app.get("/health")
async def root():
    return {"message": "API is running"}


app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(basic_router, prefix="/setup", tags=["Chat API"])
app.include_router(session_router, prefix="/sessions", tags=["Session"])
app.include_router(message_router, prefix="/messages", tags=["Message"])
app.include_router(rag_router, prefix="/rag", tags=["Rag"])





@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    try:
        create_all_tables()
        # current_state = CurrentState(current_llm=None,current_session=None)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Database init error: {str(e)}")
    yield
    logger.info("Shutting down...")


app.router.lifespan_context = lifespan




if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("src.main:app", port=port, reload=True, log_level="info")
