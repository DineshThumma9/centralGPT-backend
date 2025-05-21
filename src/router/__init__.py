# Export individual routers
from src.router.auth import router as auth_router
from src.router.basic import router as basic_router

# Create a combined router if needed
from fastapi import APIRouter

combined_router = APIRouter()
combined_router.include_router(auth_router, prefix="/auth", tags=["auth"])
combined_router.include_router(basic_router, tags=["basic"])

__all__ = ["auth_router", "basic_router", "combined_router"]