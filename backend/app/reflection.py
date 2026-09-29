"""
The "slow brain" (design doc Phase 11 / 12).

Unlike the fast per-turn pipeline in agents/orchestrator.py, this is
meant to run *after* a reply has already been sent — expensive analysis
that doesn't need to block the user waiting for a response. In this
codebase it's invoked as a normal function call right after /chat
returns (see main.py); in a real deployment this is exactly where you'd
hand off to a background job queue (Celery + Redis, per the design doc)
instead of running it inline.

It currently does two things, both intentionally simple and inspectable
rather than ML-driven, with the ML-driven version noted as the natural
V2 upgrade:

1. Emotional trend detection — compares recent-window vs. previous-window
   negative-emotion frequency and writes an episodic memory if there's a
   sustained shift, e.g. "User has seemed increasingly stressed over the
   last several conversations." This is literally frequency-counting, not
   a trained trend model — good enough to demonstrate the concept, not
   good enough to trust for anything clinical.

2. Preference contradiction flagging — if two stored "preference" memories
   look like they conflict (crude keyword-antonym check), flag both for
   human review via a `needs_review` marker in `topic` rather than
   silently picking one. Per design doc section 17 ("the user should
   remain in control of what the system believes about them"), contradictions
   are surfaced, never auto-resolved.
"""
from __future__ import annotations

import datetime
import json

from sqlalchemy.orm import Session

from app.database import Memory, Message

TREND_WINDOW = 5  # messages per window
NEGATIVE_EMOTIONS = {"sadness", "anger", "fear", "anxiety", "frustration", "disgust"}

CONTRADICTION_PAIRS = [
    ("direct", "gentle"), ("direct", "sugarcoat"), ("blunt", "gentle"),
    ("concise", "detailed"), ("short", "long"),
]


def _negative_ratio(messages: list[Message]) -> float:
    if not messages:
        return 0.0
    hits = 0
    for m in messages:
        scores = m.emotion()
        dominant = max(scores, key=scores.get) if scores else "neutral"
        if dominant in NEGATIVE_EMOTIONS:
            hits += 1
    return hits / len(messages)


def detect_emotional_trend(db: Session, user_id: str) -> str | None:
    user_msgs = (
        db.query(Message)
        .join(Message.conversation)
        .filter(Message.role == "user")
        .filter(Message.conversation.has(user_id=user_id))
        .order_by(Message.created_at)
        .all()
    )
    if len(user_msgs) < TREND_WINDOW * 2:
        return None

    recent = user_msgs[-TREND_WINDOW:]
    prior = user_msgs[-TREND_WINDOW * 2:-TREND_WINDOW]

    recent_ratio = _negative_ratio(recent)
    prior_ratio = _negative_ratio(prior)

    if recent_ratio - prior_ratio >= 0.4:
        note = (
            f"User's negative-emotion frequency rose from {prior_ratio:.0%} to "
            f"{recent_ratio:.0%} across the last {TREND_WINDOW * 2} messages — "
            "worth noting, not diagnosing."
        )
        row = Memory(
            user_id=user_id, type="episodic", topic="emotional_trend",
            content=note, importance=0.6, source="reflection",
        )
        db.add(row)
        db.commit()
        return note
    return None


def flag_contradictions(db: Session, user_id: str) -> list[dict]:
    prefs = (
        db.query(Memory)
        .filter(Memory.user_id == user_id, Memory.type == "preference")
        .order_by(Memory.created_at)
        .all()
    )
    flagged = []
    for i, a in enumerate(prefs):
        for b in prefs[i + 1:]:
            a_l, b_l = a.content.lower(), b.content.lower()
            for word_a, word_b in CONTRADICTION_PAIRS:
                if (word_a in a_l and word_b in b_l) or (word_b in a_l and word_a in b_l):
                    a.topic = "needs_review"
                    b.topic = "needs_review"
                    flagged.append({"a": a.content, "b": b.content, "reason": f"{word_a} vs {word_b}"})
    if flagged:
        db.commit()
    return flagged


def run_reflection(db: Session, user_id: str) -> dict:
    """Entry point called after a chat turn. Cheap enough to run inline
    for a demo; swap for a queued background job at real scale."""
    from app.user_model.drift import detect_drift

    trend_note = detect_emotional_trend(db, user_id)
    contradictions = flag_contradictions(db, user_id)

    # Personality drift (design doc V4) — reported, never auto-applied.
    # The EWMA in model.py already adapts the estimate on its own; this
    # is purely about surfacing that a durable shift happened so the
    # person (and the UI) can see it, consistent with this project's
    # rule that the user stays in control of what the system believes
    # about them.
    drift = [f.describe() for f in detect_drift(db, user_id)]

    return {"trend_note": trend_note, "contradictions": contradictions, "drift": drift}
