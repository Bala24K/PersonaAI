"""
Memory extraction: message -> candidate memories.

Design doc Phase 2 describes this step as an LLM call ("memory
extraction -> importance score -> embedding -> storage"). That's the
right design for production. Here it's implemented as a transparent,
inspectable rule-based extractor instead, for two reasons:

1. It must work with zero external API key, so the whole memory system
   is demonstrable offline.
2. When ANTHROPIC_API_KEY *is* configured, `llm.client` is used for
   generation, and this heuristic layer still runs as a fast, free
   pre-filter — in production you'd likely keep a cheap heuristic pass
   before spending a model call on every message anyway.

The four memory types match the design doc's taxonomy:
  semantic     - durable facts about the person ("I study CSE")
  episodic     - specific events ("I got rejected today")
  preference   - likes/dislikes/communication preferences
  relationship - other people/projects that recur in their life

Each candidate gets an importance score in [0, 1]. Only items at or
above MEMORY_IMPORTANCE_THRESHOLD (config.py) get persisted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import MEMORY_IMPORTANCE_THRESHOLD

SEMANTIC_PATTERNS = [
    (r"\bi(?:'m| am)\s+(?:a|an)\s+([a-z ]{3,40})", "semantic"),
    (r"\bi\s+(?:study|major in|work as|work in)\s+([a-z ]{3,40})", "semantic"),
    (r"\bi(?:'ve| have)\s+been\s+(learning|studying|working on)\s+([a-z0-9 ]{3,40})", "semantic"),
    (r"\b(?:ever since|since)\s+i\s+started\s+(?:my\s+)?([a-z0-9 ]{3,40})", "semantic"),
    (r"\bmy\s+(?:degree|major|job|field|work|career)\s+is\s+([a-z0-9 ]{3,40})", "semantic"),
]

EPISODIC_PATTERNS = [
    r"\b(rejected|got rejected|failed|passed|got an offer|got the job|interview(ed)?|OA|quit|fired|broke up|graduated)\b",
    r"\bdidn'?t\s+(?:go the way|turn out|work out)\b",
    r"\btoday\s+didn'?t\s+go\b",
]

PREFERENCE_PATTERNS = [
    r"\bi\s+(?:hate|dislike|can't stand|love|prefer|enjoy|like)\s+([a-z0-9 ,']{2,60})",
    r"\bplease\s+(don't|do not|stop)\s+([a-z0-9 ,']{2,60})",
    r"\bremember that i\s+([a-z0-9 ,']{2,60})",
    r"\b(?:just\s+)?don'?t\s+land\s+for\s+me\b",
    r"\bnot\s+a\s+fan\s+of\s+([a-z0-9 ,']{2,60})",
    r"\bnot\s+into\s+([a-z0-9 ,']{2,60})",
]

RELATIONSHIP_PATTERNS = [
    r"\bmy\s+(friend|professor|manager|boss|coworker|partner|sister|brother|mom|dad|roommate)\s+([A-Z][a-z]+)?",
]


@dataclass
class CandidateMemory:
    type: str
    content: str
    topic: str
    importance: float


def _clip(x: float, lo=0.0, hi=1.0) -> float:
    return max(lo, min(hi, x))


def extract_memories(text: str) -> list[CandidateMemory]:
    """Run all heuristics over one message and return scored candidates."""
    if not text or not text.strip():
        return []

    lowered = text.lower()
    candidates: list[CandidateMemory] = []

    for pattern, mtype in SEMANTIC_PATTERNS:
        for m in re.finditer(pattern, lowered):
            span = m.group(0).strip()
            importance = 0.55 + 0.1 * (len(span) > 15)
            candidates.append(CandidateMemory(mtype, span, topic="identity", importance=_clip(importance)))

    for pattern in EPISODIC_PATTERNS:
        if re.search(pattern, lowered):
            importance = 0.6
            # Emotionally loaded events (rejection/failure) matter more to
            # a companion AI than neutral ones, so nudge importance up.
            if re.search(r"reject|fail|fired|broke up", lowered):
                importance = 0.75
            candidates.append(CandidateMemory("episodic", text.strip(), topic="event", importance=_clip(importance)))

    for pattern in PREFERENCE_PATTERNS:
        for m in re.finditer(pattern, lowered):
            span = m.group(0).strip()
            candidates.append(CandidateMemory("preference", span, topic="preference", importance=_clip(0.7)))

    for pattern in RELATIONSHIP_PATTERNS:
        for m in re.finditer(pattern, lowered):
            span = m.group(0).strip()
            candidates.append(CandidateMemory("relationship", span, topic="relationship", importance=_clip(0.5)))

    # De-duplicate near-identical spans from overlapping patterns.
    seen = set()
    unique = []
    for c in candidates:
        key = (c.type, c.content)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return [c for c in unique if c.importance >= MEMORY_IMPORTANCE_THRESHOLD]
