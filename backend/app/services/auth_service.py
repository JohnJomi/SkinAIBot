"""Authentication business logic."""

from sqlalchemy.ext.asyncio import AsyncSession

from ..core.security import create_access_token, hash_password, verify_password
from ..exceptions import EmailAlreadyRegisteredError, InvalidCredentialsError
from ..models import User
from ..repositories import UserRepository


class AuthService:
    def __init__(self, session: AsyncSession):
        self.repository = UserRepository(session)

    async def register(self, email: str, password: str) -> User:
        existing = await self.repository.get_by_email(email)
        if existing is not None:
            raise EmailAlreadyRegisteredError(f"Email '{email}' is already registered.")
        return await self.repository.create(email, hash_password(password))

    async def login(self, email: str, password: str) -> str:
        user = await self.repository.get_by_email(email)
        if user is None or not verify_password(password, user.hashed_password):
            raise InvalidCredentialsError("Incorrect email or password.")
        return create_access_token(subject=str(user.id))
