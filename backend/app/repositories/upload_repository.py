"""Data access for Upload records."""

from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Upload


class UploadRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        user_id: str,
        stored_filename: str,
        original_filename: str,
        content_type: str,
        size_bytes: int,
    ) -> Upload:
        upload = Upload(
            user_id=user_id,
            stored_filename=stored_filename,
            original_filename=original_filename,
            content_type=content_type,
            size_bytes=size_bytes,
        )
        self.session.add(upload)
        await self.session.commit()
        await self.session.refresh(upload)
        return upload
