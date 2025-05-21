# Export individual routers
# Create a combined router if needed
from fastapi import APIRouter

from src.router.auth import router as auth_router
from src.router.setup import router as basic_router
from src.router.sessions import router as session_router
combined_router = APIRouter()
combined_router.include_router(auth_router, prefix="/auth", tags=["auth"])
combined_router.include_router(basic_router, tags=["basic"])
combined_router.include_router(session_router,tags=["/s"])

__all__ = ["auth_router", "basic_router", "session_router",  "combined_router"]

