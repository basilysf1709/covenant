"""Configuration via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings

_DEFAULT_DATA_DIR = Path.home() / ".covenant"
_DEFAULT_DB_URL = f"sqlite+aiosqlite:///{_DEFAULT_DATA_DIR / 'covenant.db'}"


class Settings(BaseSettings):
    model_config = {"env_prefix": "COVENANT_", "env_file": ".env"}

    database_url: str = _DEFAULT_DB_URL
    llm_provider: str = "openai"  # "openai" or "anthropic"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


def get_settings() -> Settings:
    _DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return Settings()
