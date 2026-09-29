"""
Central configuration.

Everything that would change between a laptop demo and a real deployment
lives here, read from environment variables so nothing is hardcoded deep
in the codebase.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- Database -----------------------------------------------------------
# SQLite for V1. Swap this connection string for a Postgres URL
# (e.g. postgresql://user:pass@host/db) when you move to V2 — the ORM
# models don't change, only this line does.
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DATA_DIR / 'persona.db'}")

# --- LLM ------------------------------------------------------------------
# Supports ANTHROPIC_API_KEY or GEMINI_API_KEY.
# If set, the system calls the respective live LLM API.
# If not, it falls back to a deterministic template-based responder.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")


# --- Memory / embeddings --------------------------------------------------
# Real deployments would use sentence-transformers or an embeddings API.
# This sandbox has no access to model-hosting domains (e.g. huggingface.co),
# so V1 uses a hashing-trick vectorizer: deterministic, needs no download or
# fitting step, and is a drop-in swap point (see app/memory/vectorstore.py).
EMBEDDING_DIM = 512

# How many memories to retrieve per turn.
MEMORY_TOP_K = 5

# Minimum importance score (0-1) for a candidate memory to be stored at all.
MEMORY_IMPORTANCE_THRESHOLD = 0.35

# "auto" uses the LLM-based extractor (memory/llm_extraction.py) when
# ANTHROPIC_API_KEY is set and silently falls back to the regex-based
# extractor (memory/extraction.py) if the call fails or no key is
# configured. "heuristic" and "llm" force one or the other explicitly
# (useful for testing/comparing the two).
MEMORY_EXTRACTION_MODE = os.environ.get("MEMORY_EXTRACTION_MODE", "auto")

# "heuristic" (default) always runs critic.heuristic_critique() — cheap,
# zero API cost — after generation, and attempts exactly one
# regeneration (never more, to bound cost/latency) if issues are found
# AND a real API key is configured (the offline template responder has
# no way to act on critique feedback, so regeneration is skipped for
# it — see critic.py's docstring). "off" skips the critic entirely.
# "heuristic_llm" additionally runs critic.llm_critique() when the
# heuristic pass found nothing — catches mismatches heuristics
# structurally can't (e.g. technically-acknowledging-but-still-dismissive
# tone) — at the cost of one extra API call on every clean turn, which
# is why it's opt-in rather than the default.
# "always" (default) retrieves top-K memories before every LLM call,
# same as V1-V3. "agentic" instead gives the model a search_memory tool
# (app/llm/tools.py) and lets it decide whether/how many times to call
# it — see agents/orchestrator.py and llm/client.py::generate_with_tools.
# Falls back to "always" automatically if no ANTHROPIC_API_KEY is set,
# since there's no model to make the decision with.
MEMORY_MODE = os.environ.get("MEMORY_MODE", "always")

# "heuristic" (default) always runs critic.heuristic_critique() — cheap,
# zero API cost — after generation, and attempts exactly one
# regeneration (never more, to bound cost/latency) if issues are found
# AND a real API key is configured (the offline template responder has
# no way to act on critique feedback, so regeneration is skipped for
# it — see critic.py's docstring). "off" skips the critic entirely.
# "heuristic_llm" additionally runs critic.llm_critique() when the
# heuristic pass found nothing — catches mismatches heuristics
# structurally can't (e.g. technically-acknowledging-but-still-dismissive
# tone) — at the cost of one extra API call on every clean turn, which
# is why it's opt-in rather than the default.
CRITIC_MODE = os.environ.get("CRITIC_MODE", "heuristic")

# --- Background jobs & Databases ------------------------------------------
RABBITMQ_URL = os.environ.get("RABBITMQ_URL", os.environ.get("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
MONGODB_URL = os.environ.get("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "persona_ai")
REFLECTION_MODE = os.environ.get("REFLECTION_MODE", "inline")

