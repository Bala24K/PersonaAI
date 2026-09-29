"""
The dynamic user model (design doc Phase 3 / section 15).

Deliberately NOT "USER = INTJ". Every trait is a continuous value in
[0, 1] with a confidence that grows as more evidence comes in, updated
incrementally after every message rather than computed once from a
survey. This also directly supports "personality drift" (Phase V4):
because updates are an exponentially-weighted moving average, a
sustained change in behavior shifts the estimate instead of being
permanently anchored to the first impression.

TRAITS is the whole vocabulary of what the system tracks about
communication style. Extending the model to a new trait is: add one
line here, add one signal function in signals.py.
"""
from __future__ import annotations

import datetime

from sqlalchemy.orm import Session

from app.database import Trait, TraitHistory
from app.user_model.types import TraitEstimate  # noqa: F401 -- re-exported for backward compatibility

TRAITS = [
    "humor",             # jokes, memes, playful tone
    "formality",         # inverse of casual/slang language
    "directness",        # wants blunt takes vs. hedged ones
    "verbosity",         # prefers long explanations vs. short answers
    "emoji_tendency",
    "slang_tendency",
    "technical_detail",  # wants depth/precision vs. plain-language summaries
    "cliche_aversion",   # dislikes generic motivational language
]

# How much a single new observation can move the estimate. Lower = more
# stable/slow-changing personality, higher = more reactive to recent
# messages. 0.15 means ~5 messages to mostly converge on a new pattern
# rather than either being locked in after one message or never moving.
LEARNING_RATE = 0.15


def get_or_init_traits(db: Session, user_id: str) -> dict[str, Trait]:
    rows = {t.name: t for t in db.query(Trait).filter(Trait.user_id == user_id).all()}
    created = False
    for name in TRAITS:
        if name not in rows:
            row = Trait(user_id=user_id, name=name, value=0.5, confidence=0.0, evidence_count=0)
            db.add(row)
            rows[name] = row
            created = True
    if created:
        db.commit()
    return rows


def update_trait(db: Session, user_id: str, name: str, observed_value: float, weight: float = 1.0):
    """
    Push one new observation into a trait's running estimate.

    observed_value: what this single message suggests the trait should be
                    right now (0-1), from signals.py
    weight:         how strong/unambiguous this particular signal is
                    (explicit statements like "I prefer direct answers"
                    should move the estimate more than a weak stylistic cue)
    """
    row = db.query(Trait).filter(Trait.user_id == user_id, Trait.name == name).first()
    if row is None:
        row = Trait(user_id=user_id, name=name, value=0.5, confidence=0.0, evidence_count=0)
        db.add(row)

    lr = min(LEARNING_RATE * weight, 0.9)
    row.value = (1 - lr) * row.value + lr * observed_value
    row.evidence_count += 1
    # Confidence saturates toward 1 as evidence accumulates; an explicit
    # high-weight signal counts for more evidence than a faint stylistic one.
    row.confidence = 1 - 1 / (1 + row.evidence_count * max(weight, 0.3) / 4)
    row.last_updated = datetime.datetime.utcnow()

    # Append-only history, so drift is detectable later rather than the
    # estimate just silently moving — see database.py::TraitHistory and
    # user_model/drift.py.
    db.add(TraitHistory(
        user_id=user_id, name=name, value=row.value,
        confidence=row.confidence, evidence_count=row.evidence_count,
    ))
    db.commit()


def snapshot(db: Session, user_id: str) -> list[TraitEstimate]:
    rows = get_or_init_traits(db, user_id)
    return [
        TraitEstimate(name=t.name, value=round(t.value, 3),
                       confidence=round(t.confidence, 3), evidence_count=t.evidence_count)
        for t in rows.values()
    ]
