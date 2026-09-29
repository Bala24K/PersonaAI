"""Memory Architecture (Section 7 of pitch doc).

Real version: embedding model (sentence-transformers or an API embedding model) + vector
DB + reranker.

This version: TF-IDF cosine similarity in place of learned embeddings (no model-hub
access in this sandbox), combined with recency and importance the same way the doc
specifies ("not simply top-k nearest embeddings"). The retrieval *scoring function* is
the real deliverable here — swap `_similarity` for a real embedding lookup later and
everything else (scoring, filtering, reranking) stays the same.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.db import DB

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


class MemoryStore:
    def __init__(self, db: DB):
        self._db = db

    def add(self, user_id: str, kind: str, content: str, importance: float = 0.5) -> int:
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
        with self._db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE user_id = :uid ORDER BY importance DESC, created_at DESC", {"uid": user_id}
            ).fetchall()
        return [
            Memory(
                id=r["id"], kind=r["kind"], content=r["content"], importance=r["importance"],
                created_at=r["created_at"], last_accessed_at=r["last_accessed_at"],
                access_count=r["access_count"],
            )
            for r in rows
        ]

    def delete(self, user_id: str, memory_id: int) -> bool:
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

    def retrieve(self, user_id: str, query: str, k: int = 6) -> list[Memory]:
        """Embedding-ish retrieval + recency + importance + reranking, per Section 7:
        NOT simply top-k nearest neighbor.
        """
        candidates = self.all_for_user(user_id)
        if not candidates:
            return []

        texts = [c.content for c in candidates] + [query]
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(texts)
            sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        except ValueError:
            # e.g. all-stopword query; fall back to uniform similarity
            sims = np.zeros(len(candidates))

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
        return top

    @staticmethod
    def to_prompt_block(memories: list[Memory]) -> str:
        if not memories:
            return "# MEMORIES\n(none)\n"
        lines = "\n".join(f"  - [{m.kind}] {m.content}" for m in memories)
        return f"# MEMORIES (relevant to current context)\n{lines}\n"
