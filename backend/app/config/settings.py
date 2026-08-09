"""Application settings, loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/skinaibot"
    ai_service_url: str = "http://localhost:8001"

    jwt_secret_key: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    upload_dir: str = "uploads"
    max_upload_size_bytes: int = 10 * 1024 * 1024
    allowed_upload_content_types: set[str] = {"image/jpeg", "image/png", "image/webp"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
