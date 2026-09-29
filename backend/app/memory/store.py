"""
Ties SQL storage (app.database.Memory) together with the vector index
(app.memory.vectorstore) so the rest of the app has one simple interface:
save a memory, retrieve relevant memories for the current turn.

V2 change: user filtering now happens inside the vector index itself
(Chroma's `where={"user_id": ...}`) instead of computing an allowed-ids
set here and intersecting it with search results — see vectorstore.py.
"""
from __future__ import annotations

import datetime

from sqlalchemy.orm import Session

from app.config import MEMORY_EXTRACTION_MODE, MEMORY_IMPORTANCE_THRESHOLD, MEMORY_TOP_K
from app.database import Memory
from app.memory.extraction import extract_memories
from app.memory.llm_extraction import extract_memories_llm
from app.memory.vectorstore import memory_index


def _get_candidates(text: str, source: str) -> list:
    """
    Chooses the extractor per MEMORY_EXTRACTION_MODE. "auto" (the
    default) tries the LLM extractor first and falls back to the
    heuristic one if it returns None (no API key, call failed, or
    response wasn't valid JSON) — see memory/llm_extraction.py's
    docstring for why None specifically (not an empty list) is the
    fallback signal.

    Profile imports (source="profile_import") always use the heuristic
    extractor regardless of mode: those come in as many short line-by-line
    chunks (see main.py's /profile/import), and firing one LLM call per
    line would be slow and expensive for what's usually simple factual
    text the regex patterns already handle well.

    The importance threshold is applied here, uniformly, regardless of
    which extractor ran — the heuristic extractor also filters
    internally (see extraction.py), so this is belt-and-suspenders for
    that path and the only enforcement point for the LLM path, which
    has no such filter of its own.
    """
    if source == "profile_import":
        candidates = extract_memories(text)
    elif MEMORY_EXTRACTION_MODE == "heuristic":
        candidates = extract_memories(text)
    elif MEMORY_EXTRACTION_MODE == "llm":
        result = extract_memories_llm(text)
        candidates = result if result is not None else []
    else:  # "auto"
        result = extract_memories_llm(text)
        candidates = result if result is not None else extract_memories(text)

    return [c for c in candidates if c.importance >= MEMORY_IMPORTANCE_THRESHOLD]


def ingest_text(db: Session, user_id: str, text: str, source: str = "conversation") -> list[Memory]:
    """
    Run extraction over `text` and persist any candidate memories that
    clear the importance bar. Returns the rows that were created.
    Used both for normal chat turns and for bulk profile import
    (design doc Phase 4) — same pipeline, different `source` tag.
    """
    created: list[Memory] = []
    for cand in _get_candidates(text, source):
        row = Memory(
            user_id=user_id,
            type=cand.type,
            topic=cand.topic,
            content=cand.content,
            importance=cand.importance,
            source=source,
        )
        db.add(row)
        db.flush()  # get row.id without committing yet
        memory_index.add(row.id, cand.content, user_id=user_id)
        created.append(row)
    if created:
        db.commit()
    return created


def load_index_from_db(db: Session):
    """
    Rebuild the in-process vector index from SQL on startup — only
    needed for the non-persistent fallback index (see
    vectorstore.NumpyVectorIndex.PERSISTENT). Chroma already has this
    data on disk from previous runs, so re-ingesting it every boot would
    just be redundant writes.
    """
    if getattr(memory_index, "PERSISTENT", False):
        return
    for row in db.query(Memory).all():
        memory_index.add(row.id, row.content, user_id=row.user_id)


def retrieve_relevant(db: Session, user_id: str, query: str, top_k: int = MEMORY_TOP_K) -> list[Memory]:
    """Semantic search restricted to this user's memories, most similar first."""
    hits = memory_index.search(query, top_k=top_k, user_id=user_id)
    if not hits:
        return []

    ids_in_order = [h[0] for h in hits]
    rows = db.query(Memory).filter(Memory.id.in_(ids_in_order)).all()
    rows_by_id = {r.id: r for r in rows}

    ordered = []
    for mid, _score in hits:
        row = rows_by_id.get(mid)
        if row is None:
            continue
        row.access_count = (row.access_count or 0) + 1
        row.last_accessed = datetime.datetime.utcnow()
        ordered.append(row)
    db.commit()
    return ordered
