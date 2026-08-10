"""Upload business logic."""

import io
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..exceptions import FileTooLargeError, InvalidFileTypeError
from ..models import Upload
from ..repositories import UploadRepository
from ..storage import LocalFileStorage

# Canonical MIME type per detected image format. The detected value is what we
# record, so a client cannot mislabel a PNG as something else.
FORMAT_CONTENT_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}

# Bounded so a large upload is never fully materialised in memory.
UPLOAD_CHUNK_SIZE = 64 * 1024


def detect_image_format(source: bytes | Path) -> str:
    """Return the image format decoded from the file's own bytes.

    The multipart Content-Type and filename are attacker-controlled, so they are
    not consulted. Raises InvalidFileTypeError if the content is not a supported
    image. Accepts raw bytes or a path, so a streamed upload can be validated
    without being read into memory.
    """
    settings = get_settings()
    allowed = settings.allowed_upload_content_types
    opened = io.BytesIO(source) if isinstance(source, bytes) else source

    try:
        with Image.open(opened) as image:
            # verify() checks structural integrity without decoding pixels, so
            # header-only forgeries are rejected too.
            detected_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidFileTypeError(
            "Uploaded file is not a readable image. "
            f"Allowed types: {', '.join(sorted(allowed))}."
        ) from exc

    content_type = FORMAT_CONTENT_TYPES.get(detected_format or "")
    if content_type is None or content_type not in allowed:
        raise InvalidFileTypeError(
            f"Unsupported image format '{detected_format}'. "
            f"Allowed types: {', '.join(sorted(allowed))}."
        )
    return detected_format


class UploadService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = UploadRepository(session)
        self.storage = LocalFileStorage()

    async def _stream_to_temp(self, file: UploadFile, temp_path: Path) -> int:
        """Copy the upload to temp_path in chunks, aborting once it is too large."""
        max_bytes = get_settings().max_upload_size_bytes
        size = 0
        with open(temp_path, "wb") as buffer:
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                size += len(chunk)
                if size > max_bytes:
                    # Stop reading immediately rather than draining the request.
                    raise FileTooLargeError(
                        f"File exceeds the maximum allowed size of {max_bytes} bytes."
                    )
                buffer.write(chunk)
        return size

    async def upload_image(self, user_id: str, file: UploadFile) -> Upload:
        temp_path = self.storage.new_temp_path()
        try:
            size_bytes = await self._stream_to_temp(file, temp_path)
            image_format = detect_image_format(temp_path)
            stored_filename = self.storage.promote(temp_path, image_format)
        except BaseException:
            self.storage.discard(temp_path)
            raise

        try:
            return await self.repository.create(
                user_id=user_id,
                stored_filename=stored_filename,
                original_filename=file.filename or "upload",
                content_type=FORMAT_CONTENT_TYPES[image_format],
                size_bytes=size_bytes,
            )
        except BaseException as exc:
            if self._insert_definitely_failed(exc):
                # Nothing references this file, so drop it rather than leaking it.
                self.storage.delete(stored_filename)
            raise

    def _insert_definitely_failed(self, exc: BaseException) -> bool:
        """Whether it is safe to delete the stored file after a failed insert.

        Deleting is only safe when we know no committed row points at the file.
        Getting this wrong in the other direction is worse than leaking a file:
        an orphaned file is invisible to users and reclaimable, whereas a
        committed row whose file was deleted is a permanently broken record.
        """
        if isinstance(exc, Exception):
            # A database error rolls the transaction back, so nothing persisted.
            return True
        # Cancellation (or another BaseException) delivered before COMMIT was
        # issued cannot have persisted anything either. Once COMMIT is in
        # flight the outcome is unknowable from here, so keep the file.
        return not self.repository.commit_started
