"""
LLM-based memory extraction (design doc Phase 2's actual intended
design: "memory extraction -> importance score -> embedding ->
storage" as a model call, not a regex pass).

`extraction.py` (V1/V2) is a heuristic pre-filter that works with zero
API key. This module is the upgrade Phase 2 originally called for: it
prompts the model for structured JSON output, validates the response
strictly, and returns the same `CandidateMemory` objects the heuristic
extractor does — so `memory/store.py` doesn't need to know or care
which extractor produced them.

Why this stays separate from the heuristic rather than replacing it:
1. It costs a real API call per message; the heuristic is free and
   instant, which matters for a companion app that's called on every
   turn.
2. `llm.client.complete()` returns None on any failure (no API key, rate
   limit, malformed JSON, timeout) — this module always has somewhere
   safe to fall back to.

`memory/store.py::ingest_text` decides which extractor to use per
config.MEMORY_EXTRACTION_MODE ("auto" | "heuristic" | "llm"); "auto"
uses this module when an API key is configured and silently falls back
to the heuristic otherwise, so the choice of extractor is invisible to
callers either way.
"""
from __future__ import annotations

import json
import re

from app.llm.client import complete
from app.memory.extraction import CandidateMemory

ALLOWED_TYPES = {"semantic", "episodic", "preference", "relationship"}

SYSTEM_PROMPT = """You extract durable memories from a single message a person sent to a companion AI that remembers them over time.

Return ONLY a JSON array (no prose, no markdown code fences, nothing before or after it) of objects. Each object must have exactly these fields:
- "type": one of "semantic" (a durable fact about who they are), "episodic" (a specific event that happened), "preference" (a like/dislike/communication preference), or "relationship" (a recurring person or ongoing project in their life)
- "content": a short, self-contained statement of the memory, written so it makes sense without the original message
- "topic": a short lowercase tag such as "identity", "event", "preference", or "relationship"
- "importance": a number from 0 to 1 for how worth remembering this is for future conversations

Only extract things genuinely worth remembering long-term. Do not extract greetings, small talk, or anything trivial. If the message has nothing worth remembering, return an empty array: []

Do not invent facts that aren't in the message. Do not add commentary. Output must be valid JSON and nothing else."""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _extract_json_array(raw: str) -> str | None:
    """The model is asked for pure JSON, but strip any stray wrapping
    (markdown fences, a leading 'Here is the JSON:') defensively rather
    than trusting compliance."""
    match = _JSON_ARRAY_RE.search(raw)
    return match.group(0) if match else None


def _validate_item(item: dict) -> CandidateMemory | None:
    if not isinstance(item, dict):
        return None
    mtype = item.get("type")
    content = item.get("content")
    topic = item.get("topic", "general")
    importance = item.get("importance")

    if mtype not in ALLOWED_TYPES:
        return None
    if not isinstance(content, str) or not content.strip():
        return None
    if not isinstance(topic, str):
        topic = "general"
    try:
        importance = float(importance)
    except (TypeError, ValueError):
        return None
    importance = max(0.0, min(1.0, importance))

    return CandidateMemory(type=mtype, content=content.strip(), topic=topic.strip() or "general", importance=importance)


def extract_memories_llm(text: str) -> list[CandidateMemory] | None:
    """
    Returns a list of validated CandidateMemory objects, an empty list
    if the model genuinely found nothing worth remembering, or None if
    the call failed / the response couldn't be parsed as valid JSON —
    None is the signal to the caller to fall back to the heuristic
    extractor rather than silently losing this message's memories.
    """
    if not text or not text.strip():
        return []

    raw = complete(SYSTEM_PROMPT, text, max_tokens=500)
    if raw is None:
        return None

    json_str = _extract_json_array(raw)
    if json_str is None:
        return None

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list):
        return None

    candidates = []
    for item in parsed:
        cand = _validate_item(item)
        if cand is not None:
            candidates.append(cand)
    return candidates
