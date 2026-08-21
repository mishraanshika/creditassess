"""Application settings (12-factor: everything overridable by environment)."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore",
                                      protected_namespaces=("settings_",))

    APP_NAME: str = "CreditAssess API"
    APP_VERSION: str = "1.0.0"
    ENV: str = "development"
    DEBUG: bool = True

    # --- persistence -------------------------------------------------------
    # The API runs fully without Postgres (in-memory ledger) so the demo never
    # blocks on infrastructure; set PERSISTENCE_REQUIRED=true in production.
    DATABASE_URL: str = (
        "postgresql+psycopg2://credit:credit@localhost:5432/credit_intelligence"
    )
    PERSISTENCE_REQUIRED: bool = False
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    # --- vector layer ------------------------------------------------------
    VECTOR_BACKEND: str = "faiss"       # faiss | pgvector | numpy
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 25
    # Candidates fetched per requested neighbour before the numeric re-rank.
    RERANK_POOL_FACTOR: int = 20

    # --- LLM layer ---------------------------------------------------------
    # Without a key the engine still produces a full underwriting report from a
    # deterministic template, so explainability never depends on a paid API.
    GEMINI_API_KEY: str = ""
    LLM_MODEL: str = "gemini-3.6-flash"
    LLM_MAX_TOKENS: int = 8192
    # gemini-3.x reasons before answering; thinking tokens are billed against
    # max_output_tokens, so 'low' keeps the JSON body from being truncated.
    LLM_THINKING_LEVEL: str = "low"
    LLM_TEMPERATURE: float = 0.2
    LLM_TIMEOUT_SECONDS: float = 45.0

    # --- API ---------------------------------------------------------------
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:3030", "http://127.0.0.1:3030",
    ]
    API_PREFIX: str = ""
    RATE_LIMIT_PER_MINUTE: int = 120

    # --- demo ---------------------------------------------------------------
    # Scores a couple of representative applicants at start-up so the dashboard,
    # audit trail and review queue are populated the moment the UI loads.
    DEMO_SEED: bool = True

    @property
    def sync_dsn(self) -> str:
        """psycopg2-style DSN for the raw pgvector queries."""
        return self.DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
