"""Dialogue Planner (Section 9 of pitch doc). Assembles character + user model +
relationship + EQ state + memories + scene state + conversation into the system prompt
the Main LLM actually sees.
"""
from __future__ import annotations

from app.character import Character
from app.eq_estimator import EQState
from app.memory import Memory, MemoryStore
from app.scene_state import SceneState
from app.user_state import RelationshipState, UserProfile

BASE_INSTRUCTIONS = """You are generating ONE in-character response as the character
defined below. Stay fully in character. Use the user model, relationship state, emotional
state, and memories to shape tone and content — but only reference memories that are
explicitly listed; never invent shared history that isn't there. Respond naturally, the
way the character actually would — do not narrate your reasoning, do not add disclaimers
unless the character would genuinely add them."""


def build_system_prompt(
    character: Character,
    user_profile: UserProfile,
    relationship: RelationshipState,
    eq_state: EQState,
    memories: list[Memory],
    scene: SceneState,
    eval_feedback: str | None = None,
) -> str:
    blocks = [
        BASE_INSTRUCTIONS,
        character.to_system_block(),
        user_profile.to_prompt_block(),
        relationship.to_prompt_block(),
        eq_state.to_prompt_block(),
        MemoryStore.to_prompt_block(memories),
        scene.to_prompt_block(),
    ]
    if eval_feedback:
        blocks.append(f"# REGENERATION FEEDBACK\nYour previous attempt was rejected: {eval_feedback}\n"
                       f"Fix this specific issue in your new response.\n")
    return "\n".join(b for b in blocks if b)
