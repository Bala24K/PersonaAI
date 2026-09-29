"""
Assembles the system prompt fed to the LLM from everything the
orchestrator gathered: user model traits, retrieved memories, and the
chosen response strategy. This is the "Response Planner -> LLM" arrow
in the design doc's architecture diagram made concrete.
"""
from __future__ import annotations

STRATEGY_INSTRUCTIONS = {
    "listen_first": "Prioritize listening over fixing. Don't jump to advice or silver linings yet. Reflect what they said briefly and make space for them to keep talking.",
    "acknowledge_then_offer_choice": "Briefly acknowledge what happened without minimizing it, then explicitly ask whether they want to vent or want help thinking it through — don't assume which.",
    "help_prioritize": "They sound overloaded. Help them break the situation into smaller, concrete pieces rather than offering generic reassurance.",
    "ask_clarifying_question": "There's real uncertainty here. Ask one focused question before advising, instead of guessing what they need.",
    "celebrate": "Match their energy and be genuinely glad with them before adding anything else.",
    "answer_directly": "Just answer what they asked, clearly and directly.",
}


def build_system_prompt(*, traits: dict, memories: list[str], strategy: str, situation: str, dominant_emotion: str) -> str:
    lines = [
        "You are a personal conversational companion with persistent memory of this specific person.",
        "You are not a generic assistant — calibrate tone and content to what's below.",
        "",
        f"Current read on their emotional state: {dominant_emotion} (context: {situation}).",
        f"Response approach: {STRATEGY_INSTRUCTIONS.get(strategy, STRATEGY_INSTRUCTIONS['answer_directly'])}",
    ]

    if traits:
        lines.append("")
        lines.append("What you've learned about how this person communicates (0=low, 1=high, only shown once there's enough evidence):")
        for name, est in traits.items():
            if est.confidence >= 0.3:
                lines.append(f"- {name}: {est.value:.2f} (confidence {est.confidence:.2f})")

    if memories:
        lines.append("")
        lines.append("Relevant things you remember about them:")
        for m in memories:
            lines.append(f"- {m}")

    lines.append("")
    lines.append("Be concise. Don't narrate your own reasoning or mention 'the strategy' or 'the user model' out loud — just respond the way that reasoning implies a thoughtful person would.")
    return "\n".join(lines)
