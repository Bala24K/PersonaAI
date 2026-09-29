"""Persistent User Model (Section 6 of pitch doc).

Explicitly NOT a fine-tune of the LLM after every conversation — a structured state
object updated incrementally and injected into the prompt. Relationship values are
internal system variables used to shape tone/behavior, not claims about measuring an
actual human relationship.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from app.db import DB
from app.eq_estimator import EQState


@dataclass
class RelationshipState:
    familiarity: float = 0.05
    trust: float = 0.3
    affection: float = 0.1
    playfulness: float = 0.3
    recent_tension: float = 0.0
    shared_history_summary: str = ""

    def to_prompt_block(self) -> str:
        return (
            f"# RELATIONSHIP STATE\n"
            f"Familiarity: {self.familiarity:.2f} (0=strangers, 1=very close)\n"
            f"Trust: {self.trust:.2f}\n"
            f"Affection: {self.affection:.2f}\n"
            f"Playfulness (current tone latitude): {self.playfulness:.2f}\n"
            f"Recent tension: {self.recent_tension:.2f}\n"
            f"Shared history summary: {self.shared_history_summary or '(none yet)'}\n"
        )


@dataclass
class UserProfile:
    preferences: dict = field(default_factory=dict)
    communication_preferences: dict = field(default_factory=dict)
    recurring_interests: list[str] = field(default_factory=list)
    important_facts: list[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        facts = "\n".join(f"  - {f}" for f in self.important_facts) or "  - (none recorded yet)"
        interests = ", ".join(self.recurring_interests) or "(none recorded yet)"
        return (
            f"# USER MODEL\n"
            f"Recurring interests: {interests}\n"
            f"Communication preferences: {self.communication_preferences or '(none recorded yet)'}\n"
            f"Important facts:\n{facts}\n"
        )


class UserStateStore:
    def __init__(self, db: DB):
        self._db = db

    def load(self, user_id: str) -> tuple[UserProfile, RelationshipState]:
        self._db.ensure_user(user_id)
        with self._db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_state WHERE user_id = :uid", {"uid": user_id}
            ).fetchone()
        profile_data = json.loads(row["profile_json"])
        rel_data = json.loads(row["relationship_json"])
        profile = UserProfile(
            preferences=profile_data.get("preferences", {}),
            communication_preferences=profile_data.get("communication_preferences", {}),
            recurring_interests=profile_data.get("recurring_interests", []),
            important_facts=profile_data.get("important_facts", []),
        )
        relationship = RelationshipState(**rel_data) if rel_data else RelationshipState()
        return profile, relationship

    def save(self, user_id: str, profile: UserProfile, relationship: RelationshipState):
        with self._db.connect() as conn:
            conn.execute(
                "UPDATE user_state SET profile_json = :profile, relationship_json = :rel, "
                "updated_at = :ts WHERE user_id = :uid",
                {
                    "profile": json.dumps(
                        {
                            "preferences": profile.preferences,
                            "communication_preferences": profile.communication_preferences,
                            "recurring_interests": profile.recurring_interests,
                            "important_facts": profile.important_facts,
                        }
                    ),
                    "rel": json.dumps(relationship.__dict__),
                    "ts": time.time(),
                    "uid": user_id,
                },
            )

    def apply_turn_update(
        self,
        user_id: str,
        profile: UserProfile,
        relationship: RelationshipState,
        eq_state: EQState,
        turn_count_this_session: int,
    ) -> tuple[UserProfile, RelationshipState]:
        """Small, bounded, deterministic nudges each turn — not a full re-derivation.

        Familiarity grows slowly and monotonically (capped). Trust responds to sustained
        low irritation. Tension tracks irritation with decay. This keeps relationship
        state stable rather than whipsawing on a single message, per Section 6's framing
        of these as slow-moving internal variables.
        """
        irritation = relationship.recent_tension * 0.5 + eq_state.social_state.get("irritation", 0.0) * 0.5
        relationship.recent_tension = round(min(max(irritation, 0.0), 1.0), 3)

        familiarity_gain = 0.01 if turn_count_this_session <= 40 else 0.002
        relationship.familiarity = round(min(relationship.familiarity + familiarity_gain, 1.0), 3)

        trust_delta = 0.01 if eq_state.social_state.get("irritation", 0.0) < 0.3 else -0.02
        relationship.trust = round(min(max(relationship.trust + trust_delta, 0.0), 1.0), 3)

        if eq_state.intent in ("sharing", "venting") and eq_state.intensity > 0.4:
            relationship.affection = round(min(relationship.affection + 0.005, 1.0), 3)

        return profile, relationship
