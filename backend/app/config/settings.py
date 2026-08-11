"""Application settings, loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/skinaibot"
    ai_service_url: str = "http://localhost:8001"

    # No default: the signing secret must be supplied by the environment, so a
    # misconfigured deployment fails at startup rather than signing tokens with
    # a publicly known key.
    jwt_secret_key: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    # Two bases, because two different clients fetch the same stored file and
    # they do not resolve the same hostnames.
    #
    # internal_base_url: used server-side by the AI service, which lives on the
    #   Compose network. There "localhost" is the AI container itself, so this
    #   must be the backend's service name.
    # browser_base_url: used by the user's browser, which is outside that
    #   network and cannot resolve a Compose service name at all.
    #
    # They are equal outside Docker, which is why one setting was enough until
    # the stack was containerised.
    internal_base_url: str = "http://localhost:8000"
    browser_base_url: str = "http://localhost:8000"

    upload_dir: str = "uploads"
    max_upload_size_bytes: int = 10 * 1024 * 1024
    allowed_upload_content_types: set[str] = {"image/jpeg", "image/png", "image/webp"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
