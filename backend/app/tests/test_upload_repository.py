import pytest

from app.repositories import UploadRepository


class _RecordingSession:
    """Records the order of session calls without touching a database."""

    def __init__(self):
        self.calls: list[str] = []

    def add(self, instance):
        self.calls.append("add")

    async def flush(self):
        self.calls.append("flush")

    async def refresh(self, instance):
        self.calls.append("refresh")

    async def commit(self):
        self.calls.append("commit")


@pytest.mark.asyncio
async def test_commit_is_the_final_await():
    """No await may follow COMMIT.

    A refresh after commit creates a window where cancellation arrives with the
    row already durable, which previously made the caller delete a file that a
    committed row referenced.
    """
    session = _RecordingSession()
    repository = UploadRepository(session)

    await repository.create(
        user_id="user-1",
        stored_filename="abc.png",
        original_filename="photo.png",
        content_type="image/png",
        size_bytes=123,
    )

    assert session.calls == ["add", "flush", "refresh", "commit"]
    assert session.calls[-1] == "commit"


@pytest.mark.asyncio
async def test_commit_started_is_false_until_commit_is_issued():
    session = _RecordingSession()
    repository = UploadRepository(session)
    assert repository.commit_started is False

    observed: list[bool] = []
    original_flush = session.flush

    async def flush_and_observe():
        observed.append(repository.commit_started)
        await original_flush()

    session.flush = flush_and_observe

    await repository.create(
        user_id="user-1",
        stored_filename="abc.png",
        original_filename="photo.png",
        content_type="image/png",
        size_bytes=123,
    )

    # Still false while the insert is only flushed, true once committed.
    assert observed == [False]
    assert repository.commit_started is True
