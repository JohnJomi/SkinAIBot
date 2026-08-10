"""Password hashing and JWT token helpers."""

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from ..config import get_settings
from ..exceptions import PasswordTooLongError

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# bcrypt only considers the first 72 bytes of its input. passlib truncates
# silently, which would make "<72 bytes>" and "<72 bytes><anything>" the same
# password, so reject over-long input instead of hashing a truncated prefix.
MAX_PASSWORD_BYTES = 72


def password_byte_length(password: str) -> int:
    return len(password.encode("utf-8"))


def hash_password(password: str) -> str:
    if password_byte_length(password) > MAX_PASSWORD_BYTES:
        raise PasswordTooLongError(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes when UTF-8 encoded."
        )
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    # An over-long password can never be the one we hashed, and letting passlib
    # truncate here would accept any suffix appended to a valid 72-byte password.
    if password_byte_length(plain_password) > MAX_PASSWORD_BYTES:
        return False
    return _pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> str | None:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return None
    return payload.get("sub")
