import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .ai import get_ai_client
from .ai.schemas import (
    AIAnalyzeRequest,
    AIAnalyzeResponse,
    AIChatRequest,
    AIChatResponse,
)
from .api.v1.routers import auth_router, uploads_router
from .config import get_settings
from .database.session import get_session_factory
from .exceptions import register_exception_handlers
from .repositories import UploadRepository
from .services.upload_service import reconcile_orphaned_files
from .storage import LocalFileStorage

logger = logging.getLogger(__name__)

# Resolve settings eagerly: configuration is only read lazily elsewhere, so
# without this a deployment missing JWT_SECRET_KEY would boot successfully and
# fail with a 500 on the first login instead of refusing to start.
get_settings()

app = FastAPI(
    title="Skin Disease Diagnosis API",
    description="Backend API for the Skin Disease Diagnosis AI Platform.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(auth_router)
app.include_router(uploads_router)


@app.on_event("startup")
async def _reconcile_uploads() -> None:
    """Remove stored files not referenced by any Upload row."""
    try:
        storage = LocalFileStorage()
        session_factory = get_session_factory()
        async with session_factory() as session:
            removed = await reconcile_orphaned_files(
                storage, UploadRepository(session)
            )
        if removed:
            logger.info("Startup reconciliation removed %d orphaned file(s).", len(removed))
    except Exception:
        logger.warning("Upload reconciliation skipped (database unavailable).", exc_info=True)


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "service": "Skin Disease Diagnosis API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "version": "0.1.0"}


@app.post("/api/v1/analyze", response_model=AIAnalyzeResponse)
async def analyze_image_endpoint(req: AIAnalyzeRequest):
    client = get_ai_client()
    return await client.analyze(req)

@app.post("/api/v1/chat", response_model=AIChatResponse)
async def chat_endpoint(req: AIChatRequest):
    client = get_ai_client()
    return await client.chat(req)
