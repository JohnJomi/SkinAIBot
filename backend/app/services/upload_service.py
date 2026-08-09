"""Upload business logic."""

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..exceptions import FileTooLargeError, InvalidFileTypeError
from ..models import Upload
from ..repositories import UploadRepository
from ..storage import LocalFileStorage


class UploadService:
    def __init__(self, session: AsyncSession):
        self.repository = UploadRepository(session)
        self.storage = LocalFileStorage()

    async def upload_image(self, user_id: str, file: UploadFile) -> Upload:
        settings = get_settings()

        if file.content_type not in settings.allowed_upload_content_types:
            raise InvalidFileTypeError(
                f"Unsupported file type '{file.content_type}'. "
                f"Allowed types: {', '.join(sorted(settings.allowed_upload_content_types))}."
            )

        content = await file.read()
        if len(content) > settings.max_upload_size_bytes:
            raise FileTooLargeError(
                f"File exceeds the maximum allowed size of {settings.max_upload_size_bytes} bytes."
            )

        stored_filename = self.storage.save(file.filename or "upload", content)
        return await self.repository.create(
            user_id=user_id,
            stored_filename=stored_filename,
            original_filename=file.filename or "upload",
            content_type=file.content_type or "application/octet-stream",
            size_bytes=len(content),
        )
