"""Dialogue Planner (Section 9 of pitch doc). Assembles character + user model +
relationship + EQ state + memories + scene state + conversation into the system prompt
the Main LLM actually sees.

Production hardening:
- Uses versioned prompt templates for traceability
- Clean separation of system instructions, retrieved context, and user content
- System instructions are explicitly demarcated to prevent prompt injection
- Prompt version visible in evaluation output
"""
from __future__ import annotations

from app.character import Character
from app.eq_estimator import EQState
from app.memory import Memory, MemoryStore
from app.prompts import prompt_registry, DIALOGUE_SYSTEM_PROMPT
from app.scene_state import SceneState
from app.user_state import RelationshipState, UserProfile

# Keep the base instructions for backward compatibility but source from versioned template
BASE_INSTRUCTIONS = DIALOGUE_SYSTEM_PROMPT.system_instructions


def build_system_prompt(
    character: Character,
    user_profile: UserProfile,
    relationship: RelationshipState,
    eq_state: EQState,
    memories: list[Memory],
    scene: SceneState,
    eval_feedback: str | None = None,
) -> str:
    """Build the full system prompt with versioned template and clean separation.

    Order:
    1. System instructions (protected, cannot be overridden by context)
    2. Character definition
    3. User model / relationship state / EQ state (retrieved context)
    4. Memories (retrieved context, explicitly marked as reference)
    5. Scene state (if active)
    6. Regeneration feedback (if applicable)

    The prompt version tag is included for evaluation traceability.
    """
    context_blocks = [
        character.to_system_block(),
        user_profile.to_prompt_block(),
        relationship.to_prompt_block(),
        eq_state.to_prompt_block(),
        MemoryStore.to_prompt_block(memories),
    ]

    scene_block = scene.to_prompt_block()
    if scene_block:
        context_blocks.append(scene_block)

    if eval_feedback:
        feedback_template = prompt_registry.get("regeneration_feedback")
        feedback_block = feedback_template.task_instructions.replace("{{feedback}}", eval_feedback)
        context_blocks.append(feedback_block)

    # Build using versioned template with clean separation
    prompt = DIALOGUE_SYSTEM_PROMPT.render(context_blocks=context_blocks)

    # Append version tag as comment for evaluation traceability
    prompt += f"\n\n# PROMPT_VERSION: {DIALOGUE_SYSTEM_PROMPT.version_tag}"

    return prompt


def get_prompt_versions() -> dict[str, str]:
    """Return all prompt versions for evaluation/audit output."""
    return prompt_registry.list_versions()
