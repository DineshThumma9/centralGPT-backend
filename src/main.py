import logging
import os
import time
import contextlib

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.db.dbs import create_all_tables
from src.router import auth_router, basic_router,session_router
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("api")

load_dotenv()

app = FastAPI(
    title="Chat API",
    description="API for chat application with multiple LLM providers",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_request_body(request: Request, call_next):
    body = await request.body()
    logger.info(f"Request body: {body.decode('utf-8')}")
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request started: {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            f"Request completed: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} - Error: {str(e)} - Time: {process_time:.3f}s")



@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"message": "API is running"}


@app.get("/home")
async def root_home():
    logger.info("Root endpoint called")
    return {"message": "API is running"}


app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(basic_router, prefix="/api", tags=["Chat API"])

app.include_router(session_router,prefix="/s" , tags=["Session"])


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    try:
        create_all_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")

    yield

    logger.info("Application shutting down...")


app.router.lifespan_context = lifespan

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("src.main:app", port=8001, reload=True, log_level= "info")