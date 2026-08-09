from .auth import router as auth_router
from .uploads import router as uploads_router

__all__ = ["auth_router", "uploads_router"]
