import contextlib
import os
from urllib import request

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.requests import Request

from src.db.dbs import create_all_tables
from src.router import auth_router, basic_router, session_router

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
        "http://localhost:55000"
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:55000",

    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Exception during handling {request.method} {request.url.path}: {e}", exc_info=True)
        raise
    logger.info(f"Response: {response.status_code} for {request.method} {request.url.path}")
    return response


@app.get("/")
@app.get("/health")
async def root():
    return {"message": "API is running"}


app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(basic_router, prefix="/setup", tags=["Chat API"])
app.include_router(session_router, prefix="/sessions", tags=["Session"])


from sentence_transformers import SentenceTransformer


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    try:
        create_all_tables()
        app.state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Database init error: {str(e)}")
    yield
    logger.info("Shutting down...")


app.router.lifespan_context = lifespan




if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("src.main:app", port=port, reload=True, log_level="debug")
