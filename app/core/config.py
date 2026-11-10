from functools import lru_cache
import logging
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_SQLITE_URL = f"sqlite:///{PROJECT_ROOT / 'drawing_intelligence.db'}"


class Settings(BaseSettings):
    database_url: str = DEFAULT_SQLITE_URL
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    vision_prompt_version: str = "construction-relational-v2"
    gemini_temperature: float = 0.0
    gemini_top_p: float | None = None
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    embedding_batch_size: int = 32
    gemini_timeout_seconds: float = 60.0
    gemini_retry_attempts: int = 5
    gemini_retry_initial_delay: float = 1.0
    gemini_retry_max_delay: float = 8.0
    gemini_retry_jitter: float = 1.0
    vision_min_text_chars: int = 80
    evidence_quality_good_threshold: float = 0.62
    evidence_quality_vision_threshold: float = 0.42
    # In strict mode, a Vision-only typed value remains
    # searchable but is not asserted, even if another Vision page agrees.
    qa_vision_typed_policy: str = "strict"
    qa_max_chunks_per_page: int = 2
    qa_max_chunks_per_document: int = 3
    # Soft penalties: strong multi-chunk facts may still survive, but repeated
    # page/document chunks should not crowd out independent evidence.
    qa_page_repeat_penalty: float = 1.0
    qa_document_repeat_penalty: float = 1.5
    data_root: Path = PROJECT_ROOT / "samples" / "demo-building"
    index_path: Path = Path(".data/index.json")
    vision_cache_dir: Path = Path(".data/vision-cache")
    log_level: str = "INFO"
    model_config = SettingsConfigDict(
        env_file=PROJECT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


def _database_url_is_explicit() -> bool:
    if os.environ.get("DATABASE_URL"):
        return True
    if not PROJECT_ENV_FILE.exists():
        return False
    return any(
        line.lstrip().startswith("DATABASE_URL=")
        for line in PROJECT_ENV_FILE.read_text(encoding="utf-8").splitlines()
    )


def alembic_database_url(settings: Settings | None = None) -> str:
    """Return an Alembic-safe URL from the same Settings used by the app."""
    return (settings or get_settings()).database_url.replace("%", "%%")

@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.database_url.startswith("sqlite") and not _database_url_is_explicit():
        logging.getLogger(__name__).warning(
            "DATABASE_URL is not configured; using explicit development fallback %s",
            settings.database_url,
        )
    return settings
