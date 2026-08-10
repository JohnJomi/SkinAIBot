import asyncio
import io

import pytest
from PIL import Image

from app.config import get_settings
from app.exceptions import FileTooLargeError, InvalidFileTypeError
from app.services.upload_service import UPLOAD_CHUNK_SIZE, UploadService


def make_image_bytes(image_format: str = "PNG", size=(8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format=image_format)
    return buffer.getvalue()


class _StubUploadFile:
    """Minimal UploadFile stand-in that records how much was actually read."""

    def __init__(self, content: bytes, filename="photo.png", content_type="image/png"):
        self._stream = io.BytesIO(content)
        self.filename = filename
        self.content_type = content_type
        self.bytes_read = 0

    async def read(self, size: int = -1) -> bytes:
        chunk = self._stream.read(size)
        self.bytes_read += len(chunk)
        return chunk


class _StubUploadRepository:
    """Mirrors UploadRepository's contract: commit_started flips at COMMIT.

    `error` is raised before the commit is issued (a normal DB failure);
    `error_after_commit` is raised once the row is already durable, which is how
    a cancellation arriving mid-commit looks to the caller.
    """

    def __init__(
        self,
        error: Exception | None = None,
        error_after_commit: BaseException | None = None,
    ):
        self.error = error
        self.error_after_commit = error_after_commit
        self.commit_started = False
        self.committed_rows: list[dict] = []
        self.created = None

    async def create(self, **kwargs):
        if self.error is not None:
            raise self.error

        self.commit_started = True
        self.committed_rows.append(kwargs)

        if self.error_after_commit is not None:
            raise self.error_after_commit

        self.created = kwargs
        return kwargs


class _StubSession:
    async def rollback(self):
        pass


@pytest.fixture
def upload_env(tmp_path, monkeypatch):
    """UploadService wired to a temp upload dir and an in-memory repository."""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    get_settings.cache_clear()

    def build(
        repository_error: Exception | None = None,
        error_after_commit: BaseException | None = None,
    ) -> UploadService:
        service = UploadService(_StubSession())
        service.repository = _StubUploadRepository(repository_error, error_after_commit)
        return service

    try:
        yield build, tmp_path
    finally:
        get_settings.cache_clear()


def stored_files(upload_dir):
    return sorted(p.name for p in upload_dir.iterdir())


@pytest.mark.asyncio
async def test_valid_upload_is_stored_and_recorded(upload_env):
    build, upload_dir = upload_env
    service = build()
    content = make_image_bytes("PNG")
    upload = _StubUploadFile(content)

    record = await service.upload_image("user-1", upload)

    assert record["size_bytes"] == len(content)
    assert record["content_type"] == "image/png"
    assert record["stored_filename"].endswith(".png")
    assert stored_files(upload_dir) == [record["stored_filename"]]


@pytest.mark.asyncio
async def test_oversized_upload_aborts_before_reading_everything(upload_env, monkeypatch):
    build, upload_dir = upload_env
    max_bytes = 4 * UPLOAD_CHUNK_SIZE
    monkeypatch.setenv("MAX_UPLOAD_SIZE_BYTES", str(max_bytes))
    get_settings.cache_clear()

    service = build()
    oversized = b"\x00" * (max_bytes * 4)
    upload = _StubUploadFile(oversized)

    with pytest.raises(FileTooLargeError):
        await service.upload_image("user-1", upload)

    # Stopped early instead of buffering the whole body...
    assert upload.bytes_read <= max_bytes + UPLOAD_CHUNK_SIZE
    assert upload.bytes_read < len(oversized)
    # ...and left nothing behind, not even the partial temp file.
    assert stored_files(upload_dir) == []


@pytest.mark.asyncio
async def test_upload_exactly_at_size_limit_is_accepted(upload_env, monkeypatch):
    build, upload_dir = upload_env
    content = make_image_bytes("PNG", size=(64, 64))
    monkeypatch.setenv("MAX_UPLOAD_SIZE_BYTES", str(len(content)))
    get_settings.cache_clear()

    service = build()
    record = await service.upload_image("user-1", _StubUploadFile(content))

    assert record["size_bytes"] == len(content)
    assert stored_files(upload_dir) == [record["stored_filename"]]


@pytest.mark.asyncio
async def test_spoofed_content_type_is_rejected_and_leaves_no_file(upload_env):
    """HTML bytes sent as image/jpeg with an image filename must not be stored."""
    build, upload_dir = upload_env
    service = build()
    upload = _StubUploadFile(
        b"<html><body>definitely not an image</body></html>",
        filename="innocent.jpg",
        content_type="image/jpeg",
    )

    with pytest.raises(InvalidFileTypeError):
        await service.upload_image("user-1", upload)

    assert stored_files(upload_dir) == []


@pytest.mark.asyncio
async def test_stored_extension_ignores_hostile_filename(upload_env):
    build, upload_dir = upload_env
    service = build()
    upload = _StubUploadFile(
        make_image_bytes("PNG"), filename="../../evil.php", content_type="image/png"
    )

    record = await service.upload_image("user-1", upload)

    assert record["stored_filename"].endswith(".png")
    assert "evil" not in record["stored_filename"]
    assert ".." not in record["stored_filename"]
    # The hostile name is still recorded verbatim for display only.
    assert record["original_filename"] == "../../evil.php"
    assert stored_files(upload_dir) == [record["stored_filename"]]


@pytest.mark.asyncio
async def test_stored_file_is_removed_when_db_persistence_fails(upload_env):
    """A failed commit must not leave an orphaned file on disk."""
    build, upload_dir = upload_env
    db_error = RuntimeError("commit failed")
    service = build(repository_error=db_error)

    with pytest.raises(RuntimeError) as exc_info:
        await service.upload_image("user-1", _StubUploadFile(make_image_bytes("PNG")))

    # The original exception is preserved, not masked by cleanup.
    assert exc_info.value is db_error
    assert stored_files(upload_dir) == []


@pytest.mark.asyncio
async def test_cancellation_after_commit_keeps_file_and_row(upload_env):
    """Cancellation between COMMIT and returning must not delete a committed file.

    Regression: cleanup used to fire on any BaseException, so a client
    disconnect landing just after the commit deleted a file that a durable row
    still pointed at.
    """
    build, upload_dir = upload_env
    cancellation = asyncio.CancelledError()
    service = build(error_after_commit=cancellation)

    with pytest.raises(asyncio.CancelledError):
        await service.upload_image("user-1", _StubUploadFile(make_image_bytes("PNG")))

    # The row is committed...
    assert len(service.repository.committed_rows) == 1
    stored_filename = service.repository.committed_rows[0]["stored_filename"]
    # ...so the file it references must survive.
    assert stored_files(upload_dir) == [stored_filename]
    assert (upload_dir / stored_filename).exists()


@pytest.mark.asyncio
async def test_cancellation_before_commit_still_deletes_file(upload_env):
    """Nothing was committed, so the file is a genuine orphan and must go."""
    build, upload_dir = upload_env
    service = build(repository_error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await service.upload_image("user-1", _StubUploadFile(make_image_bytes("PNG")))

    assert service.repository.committed_rows == []
    assert stored_files(upload_dir) == []


@pytest.mark.asyncio
async def test_db_error_after_commit_keeps_file(upload_env):
    """An error during COMMIT has ambiguous outcome — the file must be kept."""
    build, upload_dir = upload_env
    service = build(error_after_commit=RuntimeError("connection reset"))

    with pytest.raises(RuntimeError):
        await service.upload_image("user-1", _StubUploadFile(make_image_bytes("PNG")))

    stored_filename = service.repository.committed_rows[0]["stored_filename"]
    assert stored_files(upload_dir) == [stored_filename]


@pytest.mark.asyncio
async def test_exception_during_commit_does_not_delete_file(upload_env):
    """Regression: an exception raised mid-COMMIT must not delete the stored file.

    The server may have committed the row even though the client got an error,
    so deleting the file would leave a broken DB record.
    """
    build, upload_dir = upload_env
    service = build(error_after_commit=OSError("connection reset by peer"))

    with pytest.raises(OSError):
        await service.upload_image("user-1", _StubUploadFile(make_image_bytes("PNG")))

    # commit_started is True, so the file must survive.
    assert service.repository.commit_started is True
    stored_filename = service.repository.committed_rows[0]["stored_filename"]
    assert stored_files(upload_dir) == [stored_filename]
    assert (upload_dir / stored_filename).exists()


@pytest.mark.asyncio
async def test_cleanup_is_resilient_when_file_already_gone(upload_env, monkeypatch):
    """Cleanup must not raise a second error and mask the real failure."""
    build, upload_dir = upload_env
    db_error = RuntimeError("commit failed")
    service = build(repository_error=db_error)

    original_promote = service.storage.promote

    def promote_then_vanish(temp_path, image_format):
        stored_name = original_promote(temp_path, image_format)
        (upload_dir / stored_name).unlink()
        return stored_name

    monkeypatch.setattr(service.storage, "promote", promote_then_vanish)

    with pytest.raises(RuntimeError) as exc_info:
        await service.upload_image("user-1", _StubUploadFile(make_image_bytes("PNG")))

    assert exc_info.value is db_error
    assert stored_files(upload_dir) == []
