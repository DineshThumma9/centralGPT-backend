from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from src.router import auth_router, basic_router
from src.db.dbs import create_all_tables
import os
import logging
import time
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("api")

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Chat API",
    description="API for chat application with multiple LLM providers",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log request
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
        raise


# Add root endpoint for testing
@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"message": "API is running"}


@app.get("/home")
async def root_home():
    logger.info("Root endpoint called")
    return {"message": "API is running"}

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(basic_router, prefix="/api", tags=["Chat API"])


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    try:
        # Create database tables
        create_all_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        # Continue anyway, might be that tables already exist


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")


if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", "8000"))

    logger.info(f"Starting server on port {port}")

    # Run the application with Uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8001, reload=True, log_level="info")
