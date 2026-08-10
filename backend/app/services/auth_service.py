"""Authentication business logic."""

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.security import create_access_token, hash_password, verify_password
from ..exceptions import EmailAlreadyRegisteredError, InvalidCredentialsError
from ..models import User
from ..repositories import UserRepository

# PostgreSQL unique_violation.
_UNIQUE_VIOLATION_SQLSTATE = "23505"


def _is_email_unique_violation(exc: IntegrityError) -> bool:
    """Distinguish a duplicate email from other integrity failures."""
    sqlstate = getattr(exc.orig, "sqlstate", None) or getattr(exc.orig, "pgcode", None)
    if sqlstate is not None:
        return sqlstate == _UNIQUE_VIOLATION_SQLSTATE
    # Driver did not expose a SQLSTATE; fall back to the constraint name.
    return "ix_users_email" in str(exc.orig)


class AuthService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = UserRepository(session)

    async def register(self, email: str, password: str) -> User:
        existing = await self.repository.get_by_email(email)
        if existing is not None:
            raise EmailAlreadyRegisteredError(f"Email '{email}' is already registered.")

        hashed_password = hash_password(password)
        try:
            return await self.repository.create(email, hashed_password)
        except IntegrityError as exc:
            # A concurrent request can insert the same email between the check
            # above and this INSERT; the unique index is what actually decides.
            await self.session.rollback()
            if _is_email_unique_violation(exc):
                raise EmailAlreadyRegisteredError(
                    f"Email '{email}' is already registered."
                ) from exc
            raise

    async def login(self, email: str, password: str) -> str:
        user = await self.repository.get_by_email(email)
        if user is None or not verify_password(password, user.hashed_password):
            raise InvalidCredentialsError("Incorrect email or password.")
        return create_access_token(subject=str(user.id))
