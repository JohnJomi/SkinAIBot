import pytest

from app.config import get_settings
from app.services.upload_service import reconcile_orphaned_files
from app.storage import LocalFileStorage


class _StubStorage:
    def __init__(self, files: set[str]):
        self._files = set(files)
        self.deleted: list[str] = []

    def list_stored_files(self) -> set[str]:
        return set(self._files)

    def delete(self, name: str) -> None:
        self.deleted.append(name)
        self._files.discard(name)


class _StubRepository:
    def __init__(self, filenames: set[str]):
        self._filenames = filenames

    async def get_all_stored_filenames(self) -> set[str]:
        return set(self._filenames)


@pytest.mark.asyncio
async def test_referenced_file_is_retained():
    storage = _StubStorage({"abc.png"})
    repo = _StubRepository({"abc.png"})

    removed = await reconcile_orphaned_files(storage, repo)

    assert removed == []
    assert storage.deleted == []
    assert "abc.png" in storage.list_stored_files()


@pytest.mark.asyncio
async def test_unreferenced_stale_file_is_removed():
    storage = _StubStorage({"abc.png", "orphan.jpg"})
    repo = _StubRepository({"abc.png"})

    removed = await reconcile_orphaned_files(storage, repo)

    assert removed == ["orphan.jpg"]
    assert "orphan.jpg" not in storage.list_stored_files()
    assert "abc.png" in storage.list_stored_files()


@pytest.mark.asyncio
async def test_unrelated_non_upload_file_is_retained():
    """list_stored_files only returns known-extension files, so .gitkeep etc.
    never appear in the set and are never deleted."""
    storage = _StubStorage({"abc.png"})
    repo = _StubRepository({"abc.png"})

    removed = await reconcile_orphaned_files(storage, repo)

    assert removed == []
    assert storage.deleted == []


@pytest.mark.asyncio
async def test_real_storage_ignores_non_upload_files(tmp_path, monkeypatch):
    """A .gitkeep or .txt in the upload dir is never considered for deletion."""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        (tmp_path / ".gitkeep").touch()
        (tmp_path / "notes.txt").write_text("keep me")
        (tmp_path / "orphan.png").write_bytes(b"fake")

        storage = LocalFileStorage()
        repo = _StubRepository(set())

        removed = await reconcile_orphaned_files(storage, repo)

        assert removed == ["orphan.png"]
        assert (tmp_path / ".gitkeep").exists()
        assert (tmp_path / "notes.txt").exists()
        assert not (tmp_path / "orphan.png").exists()
    finally:
        get_settings.cache_clear()
