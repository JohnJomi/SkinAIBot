"""Upload endpoints."""

import re

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ....database import get_session
from ....exceptions import UploadNotFoundError
from ....models import User
from ....schemas import UploadResponse
from ....services import UploadService
from ....storage import LocalFileStorage
from ...dependencies import get_current_user

router = APIRouter(prefix="/api/v1/uploads", tags=["uploads"])

# Stored names are allocated as uuid4 + a known extension (see
# LocalFileStorage.allocate_name). Matching that shape exactly means the path
# parameter can never escape the upload directory: no separators, no "..", no
# extensions the storage layer would not have produced.
STORED_FILENAME = re.compile(r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\.(jpg|png|webp)$")

MEDIA_TYPES = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    upload = await UploadService(session).upload_image(str(current_user.id), file)
    storage = LocalFileStorage()
    return UploadResponse(
        id=upload.id,
        original_filename=upload.original_filename,
        content_type=upload.content_type,
        size_bytes=upload.size_bytes,
        created_at=upload.created_at,
        analysis_image_url=storage.internal_url_for(upload.stored_filename),
        display_image_url=storage.browser_url_for(upload.stored_filename),
    )


@router.get("/{stored_filename}")
async def get_upload_file(stored_filename: str) -> FileResponse:
    """Serve a stored image so `/api/v1/analyze` has something to fetch.

    Deliberately unauthenticated: the AI service fetches this URL server-side
    with no user session, so requiring a bearer token would make the analyze
    contract unusable. The stored name is an unguessable uuid4, so the URL acts
    as the capability. That is adequate for local development and NOT adequate
    for production, which wants a signed, expiring URL.
    """
    if not STORED_FILENAME.fullmatch(stored_filename):
        raise UploadNotFoundError("Upload not found.")

    path = LocalFileStorage().path_for(stored_filename)
    if not path.is_file():
        raise UploadNotFoundError("Upload not found.")

    return FileResponse(path, media_type=MEDIA_TYPES[path.suffix])
