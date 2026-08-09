"""Upload endpoints."""

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ....database import get_session
from ....models import User
from ....schemas import UploadResponse
from ....services import UploadService
from ...dependencies import get_current_user

router = APIRouter(prefix="/api/v1/uploads", tags=["uploads"])


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    upload = await UploadService(session).upload_image(str(current_user.id), file)
    return upload
