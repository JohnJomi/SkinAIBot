"""Data access for Upload records."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Upload


class UploadRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
        # True once COMMIT has been issued. After that point the row may be
        # durable even if the request is cancelled, so callers must not assume
        # the insert was discarded. See UploadService's failure handling.
        self.commit_started = False

    async def create(
        self,
        user_id: str,
        stored_filename: str,
        original_filename: str,
        content_type: str,
        size_bytes: int,
    ) -> Upload:
        """Insert an upload row and return it fully populated.

        Server-side defaults are loaded with a flush/refresh *inside* the
        transaction so that COMMIT is the final await. Refreshing after commit
        would leave a window where cancellation arrives with the row already
        durable, and the caller would wrongly treat the insert as failed.
        """
        upload = Upload(
            user_id=user_id,
            stored_filename=stored_filename,
            original_filename=original_filename,
            content_type=content_type,
            size_bytes=size_bytes,
        )
        self.session.add(upload)
        await self.session.flush()
        # Populates server defaults (created_at/updated_at) before commit. The
        # session is created with expire_on_commit=False, so these values
        # survive the commit below.
        await self.session.refresh(upload)

        self.commit_started = True
        await self.session.commit()
        return upload

    async def get_all_stored_filenames(self) -> set[str]:
        """Return every stored_filename referenced by an Upload row."""
        result = await self.session.execute(select(Upload.stored_filename))
        return set(result.scalars().all())
