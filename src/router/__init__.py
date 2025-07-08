# Export individual routers
# Create a combined router if needed
from fastapi import APIRouter

from src.router.auth import router as auth_router
from src.router.setup import router as basic_router
from src.router.sessions import router as session_router
from src.router.messages import router as message_router
from src.router.rag import router as rag_router

combined_router = APIRouter()
combined_router.include_router(auth_router, prefix="/auth", tags=["auth"])
combined_router.include_router(basic_router, tags=["basic"])
combined_router.include_router(message_router,tags=["/messages"])
combined_router.include_router(session_router,tags=["/s"])
combined_router.include_router(rag_router,tags=["/rag"])

__all__ = ["auth_router", "basic_router", "session_router","message_router", "combined_router","rag_router"]

