"""Database URL resolution for Alembic.

Kept separate from alembic/env.py because that module runs migrations as a side
effect of import, which makes it untestable.
"""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

from ..config.settings import Settings


class MigrationSettings(BaseSettings):
    """Database configuration only.

    Migrations need database credentials but no application secrets, so this
    deliberately does not inherit the app's required JWT_SECRET_KEY. The default
    is taken from the app settings so the two cannot drift apart.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Settings.model_fields["database_url"].default


def resolve_database_url() -> str:
    """Return the raw database URL, for use when actually connecting."""
    return os.getenv("DATABASE_URL") or MigrationSettings().database_url


def escape_for_alembic_config(url: str) -> str:
    """Escape a URL for Alembic's ConfigParser-backed config.

    Alembic stores options in a ConfigParser using BasicInterpolation, which
    treats '%' as a variable marker. A percent-encoded credential such as
    'P%40ss%25word' otherwise raises ValueError: invalid interpolation syntax.
    Doubling '%' is reversed by the parser on read, so callers get the original
    URL back.
    """
    return url.replace("%", "%%")
