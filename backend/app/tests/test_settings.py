import pytest
from pydantic import ValidationError

from app.config.settings import Settings


def test_settings_require_jwt_secret_key(monkeypatch, tmp_path):
    """A deployment without JWT_SECRET_KEY must fail loudly instead of using a default."""
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        # _env_file=None so a developer's local .env cannot mask the missing value.
        Settings(_env_file=None)

    assert "jwt_secret_key" in str(exc_info.value)


def test_settings_reject_empty_jwt_secret_key(monkeypatch):
    """An empty or trivially short secret is as unsafe as a missing one."""
    monkeypatch.setenv("JWT_SECRET_KEY", "")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_accept_supplied_jwt_secret_key(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "a" * 32)

    assert Settings(_env_file=None).jwt_secret_key == "a" * 32
