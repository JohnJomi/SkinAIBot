import pytest
from alembic.config import Config

from app.database.migration_url import (
    MigrationSettings,
    escape_for_alembic_config,
    resolve_database_url,
)

# Password is "P@ss%word" percent-encoded.
ENCODED_URL = "postgresql+asyncpg://user:P%40ss%25word@host:5432/db"


def test_raw_url_is_rejected_by_alembic_config():
    """Guards the reason escaping exists: the unescaped URL raises."""
    with pytest.raises(ValueError, match="interpolation"):
        Config().set_main_option("sqlalchemy.url", ENCODED_URL)


def test_escaped_url_is_accepted_and_round_trips_unchanged():
    config = Config()
    config.set_main_option("sqlalchemy.url", escape_for_alembic_config(ENCODED_URL))

    # ConfigParser un-escapes on read, so consumers see the original URL.
    assert config.get_main_option("sqlalchemy.url") == ENCODED_URL
    assert config.get_section(config.config_ini_section, {})["sqlalchemy.url"] == ENCODED_URL


def test_escape_leaves_urls_without_percent_untouched():
    plain = "postgresql+asyncpg://user:password@postgres:5432/skinaibot"
    assert escape_for_alembic_config(plain) == plain


def test_escape_handles_literal_percent():
    assert escape_for_alembic_config("pg://u:100%@h/db") == "pg://u:100%%@h/db"


def test_resolve_database_url_prefers_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", ENCODED_URL)
    assert resolve_database_url() == ENCODED_URL


def test_resolve_database_url_does_not_require_jwt_secret(monkeypatch, tmp_path):
    """Migrations must run with database credentials alone."""
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # Run from a directory without a .env so only defaults apply.
    monkeypatch.chdir(tmp_path)

    url = resolve_database_url()

    assert url.startswith("postgresql+asyncpg://")


def test_migration_settings_default_matches_app_settings(monkeypatch, tmp_path):
    """The migration default must not drift from the application default."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)

    from app.config.settings import Settings

    assert MigrationSettings().database_url == Settings.model_fields["database_url"].default


def test_migration_settings_reads_database_url_from_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.setenv("DATABASE_URL", ENCODED_URL)
    monkeypatch.chdir(tmp_path)

    assert MigrationSettings().database_url == ENCODED_URL
