"""Centralized config, loaded from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _clean_key(val: str | None) -> str | None:
    if not val:
        return None
    v = val.strip()
    if v.startswith("your_") or "api_key_here" in v or v == "":
        return None
    return v


@dataclass(frozen=True)
class Settings:
    # Multi-provider LLM Configuration
    openrouter_api_key: str | None = _clean_key(os.getenv("OPENROUTER_API_KEY"))
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
    
    groq_api_key: str | None = _clean_key(os.getenv("GROQ_API_KEY"))
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    cohere_api_key: str | None = _clean_key(os.getenv("COHERE_API_KEY"))
    cohere_model: str = os.getenv("COHERE_MODEL", "command-r-plus")

    gemini_api_key: str | None = _clean_key(os.getenv("GEMINI_API_KEY"))
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    anthropic_api_key: str | None = _clean_key(os.getenv("ANTHROPIC_API_KEY"))
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    db_path: str = os.getenv("PERSONA_DB_PATH", "data/persona.db")
    database_url: str | None = os.getenv("DATABASE_URL") or None  # if unset, falls back to SQLite at db_path



    
    # Message broker & cache
    rabbitmq_url: str = os.getenv("RABBITMQ_URL", os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//"))
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Celery configuration
    @property
    def celery_broker(self) -> str:
        # Prefer RabbitMQ as broker if specified or in standard env; fallback to Redis for local standalone
        return os.getenv("CELERY_BROKER_URL", self.rabbitmq_url)

    @property
    def celery_backend(self) -> str:
        # Redis as Celery result backend
        return os.getenv("CELERY_RESULT_BACKEND", self.redis_url)

    # MongoDB configuration
    mongodb_url: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    mongodb_db_name: str = os.getenv("MONGODB_DB_NAME", "persona_ai")

    # Context window management
    max_recent_turns: int = int(os.getenv("MAX_RECENT_TURNS", "12"))
    max_memories_retrieved: int = int(os.getenv("MAX_MEMORIES_RETRIEVED", "6"))

    # Evaluator / regeneration
    max_regeneration_attempts: int = int(os.getenv("MAX_REGENERATION_ATTEMPTS", "2"))
    min_acceptable_score: float = float(os.getenv("MIN_ACCEPTABLE_SCORE", "0.55"))

    # LLM reliability settings
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "30"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    llm_retry_backoff_base: float = float(os.getenv("LLM_RETRY_BACKOFF_BASE", "1.5"))

    # Embedding backend: "tfidf" (default) or "sentence_transformers" (requires extra deps)
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "tfidf")


settings = Settings()
