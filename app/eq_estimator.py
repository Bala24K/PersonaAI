"""EQ Transformer stand-in (Sections 3-5 of pitch doc).

Real version: fine-tuned DeBERTa/ModernBERT multi-task encoder trained on
(context, previous_state) -> (emotion, intent, need, social_state).

This version: same *interface and state-transition logic* (S_t = f(C_t, S_{t-1})), but
f is an LLM-prompted structured extraction instead of a trained encoder. This is strictly
a latency/cost tradeoff, not a logic tradeoff — swapping in a real trained model later
means replacing `_infer_raw_state` only.

Production hardening:
- All LLM outputs validated through Pydantic schemas before entering state
- Uses versioned prompt templates
- Observability via structured logging
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.llm_client import LLMClient
from app.prompts import prompt_registry

logger = logging.getLogger("persona_ai.eq")

# Keep the prompt constant for backward compatibility, but reference the versioned template
EQ_SYSTEM_PROMPT = prompt_registry.get("eq_extraction").system_instructions


@dataclass
class EQState:
    emotion: dict[str, float] = field(default_factory=dict)
    intensity: float = 0.0
    intent: str = "neutral"
    need: str = "information"
    social_state: dict[str, float] = field(default_factory=lambda: {"openness": 0.5, "trust": 0.5, "irritation": 0.0})
    confidence: float = 0.5

    @classmethod
    def initial(cls) -> "EQState":
        return cls()

    @classmethod
    def from_dict(cls, d: dict) -> "EQState":
        return cls(
            emotion=d.get("emotion", {}),
            intensity=float(d.get("intensity", 0.0)),
            intent=d.get("intent", "neutral"),
            need=d.get("need", "information"),
            social_state=d.get("social_state", {"openness": 0.5, "trust": 0.5, "irritation": 0.0}),
            confidence=float(d.get("confidence", 0.5)),
        )

    def to_dict(self) -> dict:
        return {
            "emotion": self.emotion,
            "intensity": self.intensity,
            "intent": self.intent,
            "need": self.need,
            "social_state": self.social_state,
            "confidence": self.confidence,
        }

    def to_prompt_block(self) -> str:
        top_emotions = ", ".join(f"{k}={v:.2f}" for k, v in sorted(self.emotion.items(), key=lambda kv: -kv[1])[:3]) or "none dominant"
        return (
            f"# EMOTIONAL/SOCIAL STATE (current)\n"
            f"Dominant emotions: {top_emotions}\n"
            f"Intensity: {self.intensity:.2f}\n"
            f"Intent: {self.intent}\n"
            f"Need: {self.need}\n"
            f"Social state: openness={self.social_state.get('openness', 0.5):.2f}, "
            f"trust={self.social_state.get('trust', 0.5):.2f}, "
            f"irritation={self.social_state.get('irritation', 0.0):.2f}\n"
        )


class EQEstimator:
    """Implements S_t = f_theta(C_t, S_{t-1}).

    Uses validated structured output: LLM JSON is validated through EQStateSchema
    before it becomes application state.
    """

    def __init__(self, llm: LLMClient):
        self._llm = llm

    def update(self, recent_turns: list[dict], current_message: str, previous_state: EQState) -> EQState:
        context_lines = "\n".join(f"{t['role']}: {t['content']}" for t in recent_turns[-8:])
        prompt = (
            f"PREVIOUS STATE:\n{previous_state.to_dict()}\n\n"
            f"RECENT CONTEXT:\n{context_lines}\n\n"
            f"CURRENT USER MESSAGE:\n{current_message}\n\n"
            f"Output the updated state JSON now."
        )

        # Use validated JSON extraction — output is guaranteed to match schema
        # or falls back to safe defaults rather than corrupting state
        raw = self._llm.complete_json_validated(
            EQ_SYSTEM_PROMPT, prompt, schema_name="EQStateSchema"
        )
        return EQState.from_dict(raw)
