"""
Semantic similarity layer for memory retrieval — V2.

V1 used a flat in-memory numpy matrix, rebuilt from SQL on every process
start. V2 replaces that with Chroma, a real embedded vector database,
while keeping the same embedding function (`embed()`, still a
hashing-trick vectorizer — see the note below on why that specific piece
hasn't changed).

What actually improved by moving to Chroma:
  - Persistent, on-disk storage (data/chroma/) instead of an in-memory
    structure that has to be rebuilt from SQL on every restart.
  - Approximate nearest-neighbor search (HNSW) instead of brute-force
    cosine over a Python/numpy matrix — matters once memory counts get
    large; doesn't matter yet at demo scale, but the interface is now
    the same one you'd use against a much bigger memory store.
  - Native per-user metadata filtering (`where={"user_id": ...}`)
    instead of computing an `allowed_ids` set in Python and intersecting
    it with search results by hand.
  - This is literally one of the two vector databases the original
    design doc named (Qdrant or Chroma) — this file is that step done,
    not a bigger stand-in for it.

What did NOT change, and why: `embed()` is still the same hashing-trick
vectorizer from V1. Swapping Chroma in doesn't touch embedding quality
at all — retrieval is still lexical/n-gram similarity, not semantic
similarity, for the same reason as before (no network access to
model-hosting domains in this sandbox). That upgrade is still tracked
separately in docs/ARCHITECTURE.md as the highest-leverage next step.

If chromadb fails to import or initialize for any reason, this module
falls back to the V1 in-memory implementation automatically so the app
never hard-fails over a vector store issue — see `build_vector_index()`
at the bottom.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from app.config import DATA_DIR, EMBEDDING_DIM

_vectorizer = HashingVectorizer(
    n_features=EMBEDDING_DIM,
    alternate_sign=False,
    ngram_range=(1, 2),
    norm="l2",
)


def embed(text: str) -> np.ndarray:
    """Deterministically embed a piece of text into a fixed-size vector."""
    vec = _vectorizer.transform([text or ""]).toarray()[0]
    return vec.astype(np.float32)


class ChromaVectorIndex:
    """Persistent, per-user-filterable vector index backed by Chroma."""

    PERSISTENT = True  # data survives process restart; store.py uses this
                        # to skip re-ingesting everything from SQL on boot.

    def __init__(self, persist_path: Path):
        import chromadb  # local import so a missing/broken install doesn't break the whole app at import time

        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._collection = self._client.get_or_create_collection(
            name="memories", metadata={"hnsw:space": "cosine"}
        )

    def add(self, item_id: str, text: str, user_id: str = "default"):
        vec = embed(text).tolist()
        try:
            self._collection.add(
                ids=[item_id], embeddings=[vec], documents=[text],
                metadatas=[{"user_id": user_id}],
            )
        except Exception:
            self._collection.upsert(
                ids=[item_id], embeddings=[vec], documents=[text],
                metadatas=[{"user_id": user_id}],
            )

    def search(self, query: str, top_k: int, user_id: str = "default"):
        """Returns [(item_id, similarity_score), ...] sorted descending."""
        if self._collection.count() == 0:
            return []
        vec = embed(query).tolist()
        result = self._collection.query(
            query_embeddings=[vec], n_results=min(top_k, self._collection.count()),
            where={"user_id": user_id},
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        # hnsw:space="cosine" -> distance = 1 - cosine_similarity
        return [(i, 1.0 - d) for i, d in zip(ids, distances)]


class NumpyVectorIndex:
    """V1 fallback: flat in-memory index, used only if Chroma is unavailable."""

    PERSISTENT = False  # lost on restart; store.py rebuilds it from SQL on boot.

    def __init__(self):
        self._ids: list[str] = []
        self._users: list[str] = []
        self._vectors: np.ndarray | None = None

    def add(self, item_id: str, text: str, user_id: str = "default"):
        vec = embed(text).reshape(1, -1)
        self._vectors = vec if self._vectors is None else np.vstack([self._vectors, vec])
        self._ids.append(item_id)
        self._users.append(user_id)

    def search(self, query: str, top_k: int, user_id: str = "default"):
        if self._vectors is None or not self._ids:
            return []
        q = embed(query)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        sims = (self._vectors @ q) / (np.linalg.norm(self._vectors, axis=1) * qn + 1e-8)
        order = np.argsort(-sims)
        results = []
        for i in order:
            if self._users[i] != user_id:
                continue
            results.append((self._ids[i], float(sims[i])))
            if len(results) >= top_k:
                break
        return results


def build_vector_index():
    """Try Chroma first; fall back to the in-memory index on any failure."""
    try:
        return ChromaVectorIndex(DATA_DIR / "chroma")
    except Exception as e:  # noqa: BLE001 - deliberately broad: any failure here should degrade, not crash
        import sys
        print(f"[memory.vectorstore] Chroma unavailable ({e}); falling back to in-memory index.", file=sys.stderr)
        return NumpyVectorIndex()


# One instance for the process. Both backends expose the same
# add(id, text, user_id) / search(query, top_k, user_id) interface.
memory_index = build_vector_index()
