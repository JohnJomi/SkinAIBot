"""Pydantic schemas for User and auth."""

import uuid

from pydantic import BaseModel, EmailStr, Field, field_validator

from ..core.security import MAX_PASSWORD_BYTES, password_byte_length


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)

    @field_validator("password")
    @classmethod
    def password_within_bcrypt_limit(cls, value: str) -> str:
        # bcrypt ignores anything past 72 bytes, so reject rather than accept a
        # password whose tail would be silently discarded. The limit is bytes,
        # not characters: multibyte UTF-8 hits it sooner than len() suggests.
        if password_byte_length(value) > MAX_PASSWORD_BYTES:
            raise ValueError(
                f"Password must be at most {MAX_PASSWORD_BYTES} bytes when UTF-8 encoded."
            )
        return value


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: uuid.UUID
    email: EmailStr
    is_active: bool

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    sub: str
