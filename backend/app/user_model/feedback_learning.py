"""
Feedback -> user model loop (design doc Phase 10, "feedback learning").

Before this module, `/feedback` only stored ratings — the design doc's
"feedback -> dataset -> evaluation -> model improvement" pipeline
stopped at "dataset". This module closes the smallest useful version of
that loop: when someone down-votes a reply with a specific tag ("too
formal", "too verbose", ...), the trait that tag is actually about gets
nudged in the corrected direction immediately, using the same EWMA
update mechanism (`user_model.model.update_trait`) that ordinary message
signals use — so a down-vote is treated as one more (unusually
high-confidence, since it's explicit and deliberate) observation, not a
separate mechanism bolted on top.

Up-votes (added this round) reinforce rather than correct: they nudge
the relevant trait(s) toward their own *current* value, which is a
no-op on the value itself but raises confidence — "this setting is
working, keep it" without pretending to know a target it should move
toward. Specific positive tags ("good_formality", "good_humor", ...)
reinforce that one trait at a higher weight; an untagged or
unrecognized-tag up-vote reinforces a broader set of core traits at a
lower weight, since a plain thumbs-up is genuinely weaker evidence
about *which* trait, if any, was actually responsible for the good
reply — that credit-assignment problem doesn't go away just because
there's now a mechanism; it's just now handled by trusting specific
signals more than vague ones, which is the same principle
`update_trait`'s weight parameter already encodes everywhere else in
this codebase.

What this still deliberately does NOT do, honestly documented rather
than silently absent:
- The "wrong_memory" tag now traces back to the actual memories used
  for the message being rated (see main.py's /feedback route, which
  reads Message.retrieved_memory_ids_json) and flags them for review —
  but only for messages generated after this schema addition. Existing
  messages from before this change have no recorded memory IDs to trace
  back to, so "wrong_memory" on an old message still falls through to
  unhandled.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.user_model.model import get_or_init_traits, update_trait

# tag -> (trait_name, corrected_target_value, weight)
# The target is where the trait *should* sit given this complaint, not a
# delta — e.g. "too_formal" means formality was read too high, so nudge
# toward a clearly-lower value. Weight is high (0.75) because this is an
# explicit, deliberate correction, not an ambient stylistic cue — it
# deserves more trust than most signals in user_model/signals.py.
TAG_TRAIT_ADJUSTMENTS: dict[str, tuple[str, float, float]] = {
    "too_formal": ("formality", 0.15, 0.75),
    "too_casual": ("formality", 0.85, 0.75),
    "too_verbose": ("verbosity", 0.15, 0.75),
    "too_short": ("verbosity", 0.85, 0.75),
    "too_brief": ("verbosity", 0.85, 0.75),
    "too_blunt": ("directness", 0.2, 0.75),
    "too_vague": ("directness", 0.85, 0.75),
    "not_direct_enough": ("directness", 0.85, 0.75),
    "too_technical": ("technical_detail", 0.15, 0.75),
    "not_technical_enough": ("technical_detail", 0.85, 0.75),
    "too_cliche": ("cliche_aversion", 0.85, 0.8),
    "generic": ("cliche_aversion", 0.85, 0.8),
    "too_much_humor": ("humor", 0.15, 0.75),
    "not_enough_humor": ("humor", 0.85, 0.75),
}

# tag -> trait_name. An up-vote with one of these tags reinforces that
# specific trait toward its own current value at a higher weight than
# the generic case below — a targeted compliment is stronger evidence
# than a plain thumbs-up.
POSITIVE_TAG_TRAIT_MAP: dict[str, str] = {
    "good_formality": "formality",
    "good_verbosity": "verbosity",
    "good_directness": "directness",
    "good_technical_level": "technical_detail",
    "good_humor": "humor",
}
POSITIVE_TAG_WEIGHT = 0.6

# An untagged (or unrecognized-tag) up-vote reinforces this broader set
# at a lower weight — see module docstring on why this stays modest
# rather than pretending to solve credit assignment.
GENERIC_REINFORCE_TRAITS = ["formality", "verbosity", "directness", "humor", "technical_detail"]
GENERIC_REINFORCE_WEIGHT = 0.25

# Tags that are real and useful but aren't trait-adjustments — surfaced
# separately so callers/logs can see they were received even though this
# module can't act on them for older messages (see docstring re:
# wrong_memory tracing, which now works for new messages via main.py).
KNOWN_NON_TRAIT_TAGS = {"wrong_memory", "wrong_emotion", "unhelpful"}


def _apply_downvote(db: Session, user_id: str, tags: list[str]) -> dict:
    adjusted = []
    unhandled = []
    for tag in tags:
        if tag in TAG_TRAIT_ADJUSTMENTS:
            trait_name, target_value, weight = TAG_TRAIT_ADJUSTMENTS[tag]
            update_trait(db, user_id, trait_name, target_value, weight)
            adjusted.append({"tag": tag, "trait": trait_name, "nudged_toward": target_value})
        else:
            unhandled.append(tag)  # includes KNOWN_NON_TRAIT_TAGS and anything unrecognized
    return {"adjusted_traits": adjusted, "unhandled_tags": unhandled}


def _apply_upvote(db: Session, user_id: str, tags: list[str]) -> dict:
    current_traits = get_or_init_traits(db, user_id)
    adjusted = []
    unhandled = []
    specifically_reinforced = set()

    for tag in tags:
        if tag in POSITIVE_TAG_TRAIT_MAP:
            trait_name = POSITIVE_TAG_TRAIT_MAP[tag]
            current_value = current_traits[trait_name].value
            update_trait(db, user_id, trait_name, current_value, POSITIVE_TAG_WEIGHT)
            adjusted.append({"tag": tag, "trait": trait_name, "reinforced_at": round(current_value, 3)})
            specifically_reinforced.add(trait_name)
        else:
            unhandled.append(tag)

    # Generic reinforcement only applies to traits not already reinforced
    # specifically above, so one up-vote never counts as two stacked
    # observations for the same trait.
    for trait_name in GENERIC_REINFORCE_TRAITS:
        if trait_name in specifically_reinforced:
            continue
        current_value = current_traits[trait_name].value
        update_trait(db, user_id, trait_name, current_value, GENERIC_REINFORCE_WEIGHT)

    if not specifically_reinforced:
        adjusted.append({
            "tag": "(untagged or unrecognized)", "trait": "formality/verbosity/directness/technical_detail/humor",
            "note": "confidence reinforced at current values, nothing changed",
        })

    return {"adjusted_traits": adjusted, "unhandled_tags": unhandled}


def flag_wrong_memory(db: Session, message) -> list[str]:
    """
    Given the Message row a "wrong_memory" down-vote was about, flags
    every memory that was actually retrieved/used for it (topic set to
    "needs_review", same convention reflection.py's contradiction
    detection uses) rather than guessing or flagging nothing. Returns
    the list of memory IDs flagged.

    Only works for messages that have retrieved_memory_ids_json
    populated — i.e. generated after this feature was added. Older
    messages have "[]" there and this returns an empty list, which
    main.py surfaces honestly rather than pretending something was done.
    """
    from app.database import Memory

    ids = message.retrieved_memory_ids()
    if not ids:
        return []
    rows = db.query(Memory).filter(Memory.id.in_(ids)).all()
    for row in rows:
        row.topic = "needs_review"
    db.commit()
    return [r.id for r in rows]


def apply_feedback(db: Session, user_id: str, rating: str, tags: list[str]) -> dict:
    """
    Applies whatever trait correction (down-vote) or reinforcement
    (up-vote) a rating's tags imply. Returns a small report of what was
    adjusted vs. what was received but not actionable, for transparency
    (surfaced in the API response so this isn't a silent black box).
    """
    if rating == "down":
        return _apply_downvote(db, user_id, tags)
    if rating == "up":
        return _apply_upvote(db, user_id, tags)
    return {"adjusted_traits": [], "unhandled_tags": tags, "note": f"unrecognized rating '{rating}'"}
