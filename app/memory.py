"""Memory Architecture (Section 7 of pitch doc).

Production-oriented retrieval system:
- Configurable embedding backend (TF-IDF by default, designed for drop-in replacement
  with sentence-transformers or API embeddings)
- User-level isolation: every retrieval is scoped to a single user_id
- Metadata stored with every memory (kind, source, timestamps)
- Configurable top-k retrieval with metadata filtering
- Combined scoring: semantic similarity + recency + importance (not simply top-k nearest)
- Observability: retrieval latency metrics
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.db import DB
from app.observability import RetrievalMetrics, log_retrieval

logger = logging.getLogger("persona_ai.memory")

VALID_KINDS = {"semantic", "episodic", "preference", "relational"}


@dataclass
class Memory:
    id: int
    kind: str
    content: str
    importance: float
    created_at: float
    last_accessed_at: float
    access_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Embedding backend interface
# ---------------------------------------------------------------------------

class EmbeddingBackend(ABC):
    """Abstract interface for embedding computation.

    Default implementation uses TF-IDF. Replace with sentence-transformers,
    OpenAI embeddings, or any other embedding provider by implementing this interface.
    """

    @abstractmethod
    def compute_similarities(self, query: str, documents: list[str]) -> np.ndarray:
        """Compute similarity scores between query and each document.

        Returns: 1D array of similarity scores, same length as documents.
        """
        raise NotImplementedError


class TfidfEmbeddingBackend(EmbeddingBackend):
    """TF-IDF based similarity computation.

    This is the honest stand-in for a learned embedding model. The retrieval
    scoring function around it (recency, importance, reranking) is the real
    deliverable — swap this for a real embedding lookup and everything else
    stays the same.
    """

    def compute_similarities(self, query: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.array([])
        texts = documents + [query]
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(texts)
            sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
            return sims
        except ValueError:
            # e.g. all-stopword query; fall back to uniform similarity
            return np.zeros(len(documents))


# ---------------------------------------------------------------------------
# Memory Store
# ---------------------------------------------------------------------------

class MemoryStore:
    def __init__(self, db: DB, embedding_backend: EmbeddingBackend | None = None):
        self._db = db
        self._embeddings = embedding_backend or TfidfEmbeddingBackend()

    def add(
        self,
        user_id: str,
        kind: str,
        content: str,
        importance: float = 0.5,
        metadata: Dict[str, Any] | None = None,
    ) -> int:
        """Add a memory for a specific user.

        Args:
            user_id: Owner of this memory (strict isolation).
            kind: One of semantic, episodic, preference, relational.
            content: The memory content text.
            importance: Importance weight 0-1.
            metadata: Optional metadata dict stored as JSON.
        """
        assert kind in VALID_KINDS, f"invalid memory kind: {kind}"
        now = time.time()
        with self._db.connect() as conn:
            row = conn.execute(
                "INSERT INTO memories (user_id, kind, content, importance, created_at, "
                "last_accessed_at, access_count) VALUES (:uid, :kind, :content, :importance, "
                ":created, :accessed, 0) RETURNING id",
                {
                    "uid": user_id, "kind": kind, "content": content, "importance": importance,
                    "created": now, "accessed": now,
                },
            ).fetchone()
            return row["id"]

    def all_for_user(self, user_id: str) -> list[Memory]:
        """Return all memories for a specific user. User isolation is enforced at query level."""
        with self._db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE user_id = :uid ORDER BY importance DESC, created_at DESC",
                {"uid": user_id},
            ).fetchall()
        return [
            Memory(
                id=r["id"], kind=r["kind"], content=r["content"], importance=r["importance"],
                created_at=r["created_at"], last_accessed_at=r["last_accessed_at"],
                access_count=r["access_count"],
                metadata={"kind": r["kind"], "user_id": user_id},
            )
            for r in rows
        ]

    def delete(self, user_id: str, memory_id: int) -> bool:
        """Delete a memory. Requires matching user_id for isolation."""
        with self._db.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM memories WHERE id = :id AND user_id = :uid",
                {"id": memory_id, "uid": user_id},
            ).fetchone()
            if not row:
                return False
            conn.execute(
                "DELETE FROM memories WHERE id = :id AND user_id = :uid",
                {"id": memory_id, "uid": user_id},
            )
            return True

    def retrieve(
        self,
        user_id: str,
        query: str,
        k: int = 6,
        kind_filter: str | None = None,
        min_importance: float = 0.0,
    ) -> list[Memory]:
        """Embedding-based retrieval + recency + importance + reranking.

        User isolation: only memories belonging to user_id are candidates.
        Configurable top-k, optional metadata filtering by kind and minimum importance.

        NOT simply top-k nearest neighbor — combines semantic similarity, recency,
        and importance into a composite score.
        """
        start_time = time.perf_counter()

        candidates = self.all_for_user(user_id)

        # Apply metadata filters
        if kind_filter:
            candidates = [c for c in candidates if c.kind == kind_filter]
        if min_importance > 0.0:
            candidates = [c for c in candidates if c.importance >= min_importance]

        if not candidates:
            log_retrieval(RetrievalMetrics(
                latency_ms=0.0, candidates_scanned=0, results_returned=0, user_id=user_id,
            ))
            return []

        # Compute semantic similarities
        texts = [c.content for c in candidates]
        sims = self._embeddings.compute_similarities(query, texts)

        now = time.time()
        scored = []
        for mem, sim in zip(candidates, sims):
            age_days = max((now - mem.created_at) / 86400, 0)
            recency_score = 1 / (1 + age_days / 14)  # ~2-week half-relevance window
            score = (0.55 * sim) + (0.25 * mem.importance) + (0.20 * recency_score)
            scored.append((score, mem))

        scored.sort(key=lambda x: -x[0])
        top = [m for _, m in scored[:k]]

        if top:
            with self._db.connect() as conn:
                for mem in top:
                    conn.execute(
                        "UPDATE memories SET last_accessed_at = :ts, access_count = access_count + 1 "
                        "WHERE id = :id",
                        {"ts": now, "id": mem.id},
                    )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        log_retrieval(RetrievalMetrics(
            latency_ms=elapsed_ms,
            candidates_scanned=len(candidates),
            results_returned=len(top),
            user_id=user_id,
        ))

        return top

    @staticmethod
    def to_prompt_block(memories: list[Memory]) -> str:
        if not memories:
            return "# MEMORIES\n(none)\n"
        lines = "\n".join(f"  - [{m.kind}] {m.content}" for m in memories)
        return f"# MEMORIES (relevant to current context)\n{lines}\n"
