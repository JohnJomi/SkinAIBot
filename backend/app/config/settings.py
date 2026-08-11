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

    # Absolute base for URLs this service hands out for its own stored files.
    # It must be resolvable by whoever fetches them, which is the AI service,
    # not the browser: inside Compose that is the `backend` service name, so
    # the value is supplied per environment rather than assumed here.
    public_base_url: str = "http://localhost:8000"

    upload_dir: str = "uploads"
    max_upload_size_bytes: int = 10 * 1024 * 1024
    allowed_upload_content_types: set[str] = {"image/jpeg", "image/png", "image/webp"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
