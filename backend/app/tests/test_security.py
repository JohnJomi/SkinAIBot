import pytest
from pydantic import ValidationError

from app.core.security import (
    MAX_PASSWORD_BYTES,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.exceptions import PasswordTooLongError
from app.schemas import UserCreate

# 24 three-byte characters == exactly 72 UTF-8 bytes, but only 24 Python chars.
MULTIBYTE_AT_LIMIT = "€" * 24
MULTIBYTE_OVER_LIMIT = "€" * 25


def test_hash_password_verifies_correct_password():
    hashed = hash_password("correct-horse-battery-staple")
    assert verify_password("correct-horse-battery-staple", hashed)


def test_hash_password_rejects_incorrect_password():
    hashed = hash_password("correct-horse-battery-staple")
    assert not verify_password("wrong-password", hashed)


def test_access_token_roundtrip():
    token = create_access_token(subject="user-123")
    assert decode_access_token(token) == "user-123"


def test_decode_access_token_rejects_garbage_token():
    assert decode_access_token("not-a-real-token") is None


def test_hash_password_accepts_exactly_72_ascii_bytes():
    password = "a" * MAX_PASSWORD_BYTES
    assert verify_password(password, hash_password(password))


def test_hash_password_rejects_73_ascii_bytes():
    with pytest.raises(PasswordTooLongError):
        hash_password("a" * (MAX_PASSWORD_BYTES + 1))


def test_hash_password_accepts_exactly_72_utf8_bytes_of_multibyte_characters():
    assert len(MULTIBYTE_AT_LIMIT.encode("utf-8")) == MAX_PASSWORD_BYTES
    assert verify_password(MULTIBYTE_AT_LIMIT, hash_password(MULTIBYTE_AT_LIMIT))


def test_hash_password_rejects_multibyte_password_over_72_bytes():
    # Only 25 characters, but 75 bytes: the limit must be measured in bytes.
    assert len(MULTIBYTE_OVER_LIMIT) < MAX_PASSWORD_BYTES
    with pytest.raises(PasswordTooLongError):
        hash_password(MULTIBYTE_OVER_LIMIT)


def test_over_long_password_does_not_verify_against_truncated_prefix():
    """bcrypt truncates at 72 bytes; a longer password must not authenticate."""
    hashed = hash_password("a" * MAX_PASSWORD_BYTES)
    assert not verify_password("a" * MAX_PASSWORD_BYTES + "extra", hashed)


def test_user_create_accepts_password_at_byte_limit():
    assert UserCreate(email="user@example.com", password="a" * MAX_PASSWORD_BYTES)
    assert UserCreate(email="user@example.com", password=MULTIBYTE_AT_LIMIT)


@pytest.mark.parametrize(
    "password",
    ["a" * (MAX_PASSWORD_BYTES + 1), MULTIBYTE_OVER_LIMIT],
    ids=["73-ascii-bytes", "75-utf8-bytes"],
)
def test_user_create_rejects_password_over_byte_limit(password):
    """Registration surfaces a validation error (422), never a 500."""
    with pytest.raises(ValidationError):
        UserCreate(email="user@example.com", password=password)
