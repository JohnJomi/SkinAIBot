import pytest
from sqlalchemy.exc import IntegrityError

from app.exceptions import EmailAlreadyRegisteredError
from app.services.auth_service import AuthService


class _StubOrig(Exception):
    """Stand-in for the DBAPI error wrapped by IntegrityError."""

    def __init__(self, sqlstate: str | None, message: str = ""):
        self.sqlstate = sqlstate
        super().__init__(message or "duplicate key value violates unique constraint")


class _StubSession:
    def __init__(self):
        self.rolled_back = False

    async def rollback(self):
        self.rolled_back = True


class _StubRepository:
    """Simulates the INSERT losing a race against a concurrent registration."""

    def __init__(self, error: Exception | None):
        self.error = error

    async def get_by_email(self, email: str):
        # The pre-check passes: the competing row does not exist yet.
        return None

    async def create(self, email: str, hashed_password: str):
        raise self.error


def _service_raising(error: Exception) -> tuple[AuthService, _StubSession]:
    session = _StubSession()
    service = AuthService(session)
    service.repository = _StubRepository(error)
    return service, session


def _integrity_error(orig: Exception) -> IntegrityError:
    return IntegrityError("INSERT INTO users ...", {}, orig)


@pytest.mark.asyncio
async def test_register_translates_unique_violation_to_email_already_registered():
    """The get_by_email/INSERT race must not surface as an unhandled 500."""
    service, session = _service_raising(_integrity_error(_StubOrig(sqlstate="23505")))

    with pytest.raises(EmailAlreadyRegisteredError) as exc_info:
        await service.register("race@example.com", "a-valid-password")

    assert exc_info.value.status_code == 409
    assert session.rolled_back, "session must be rolled back before raising"


@pytest.mark.asyncio
async def test_register_falls_back_to_constraint_name_without_sqlstate():
    orig = _StubOrig(
        sqlstate=None,
        message='duplicate key value violates unique constraint "ix_users_email"',
    )
    service, session = _service_raising(_integrity_error(orig))

    with pytest.raises(EmailAlreadyRegisteredError):
        await service.register("race@example.com", "a-valid-password")

    assert session.rolled_back


@pytest.mark.asyncio
async def test_register_reraises_unrelated_integrity_errors():
    """A non-unique integrity failure must not be mislabelled as a duplicate email."""
    orig = _StubOrig(sqlstate="23502", message="null value in column violates not-null")
    service, session = _service_raising(_integrity_error(orig))

    with pytest.raises(IntegrityError):
        await service.register("someone@example.com", "a-valid-password")

    assert session.rolled_back
