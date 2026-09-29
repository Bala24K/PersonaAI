"""Response Evaluator (Section 11 of pitch doc).

Real version: trained reward/preference model.
This version: cheap heuristics (repetition, length sanity, echo-detection) combined with
an LLM self-critique call for the harder-to-heuristic dimensions (emotional fit, persona
consistency). Interface matches what a trained model would return, so it's a drop-in
replacement point later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.character import Character
from app.eq_estimator import EQState
from app.llm_client import LLMClient

EVAL_SYSTEM_PROMPT = """You are a strict response-quality evaluator for a character-based
conversational AI. Given the character definition, the user's emotional state, and a
candidate response, score the response. Output ONLY JSON:
{
  "emotional_fit": <0-1, does the tone match what the user needs right now>,
  "persona_consistency": <0-1, does this sound like the defined character>,
  "context_relevance": <0-1, does it actually respond to what the user said>,
  "memory_consistency": <0-1, does it avoid contradicting/inventing memories>,
  "repetition": <0-1, HIGH score = repetitive/generic, LOW score = fresh>
}
Be genuinely critical — most responses should NOT get near-perfect scores."""


@dataclass
class EvalResult:
    emotional_fit: float
    persona_consistency: float
    context_relevance: float
    memory_consistency: float
    repetition: float

    @property
    def overall(self) -> float:
        # repetition is inverted (high repetition = bad)
        return round(
            0.30 * self.emotional_fit
            + 0.25 * self.persona_consistency
            + 0.25 * self.context_relevance
            + 0.10 * self.memory_consistency
            + 0.10 * (1 - self.repetition),
            3,
        )

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["overall"] = self.overall
        return d


class ResponseEvaluator:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def evaluate(
        self,
        character: Character,
        eq_state: EQState,
        recent_assistant_turns: list[str],
        candidate: str,
    ) -> EvalResult:
        heuristic_repetition = self._heuristic_repetition(recent_assistant_turns, candidate)

        prompt = (
            f"CHARACTER:\n{character.to_system_block()}\n"
            f"USER EMOTIONAL STATE:\n{eq_state.to_dict()}\n"
            f"CANDIDATE RESPONSE:\n{candidate}\n"
        )
        raw = self._llm.complete_json(EVAL_SYSTEM_PROMPT, prompt)

        # blend heuristic repetition with model-judged repetition (max = more conservative)
        repetition = max(float(raw.get("repetition", 0.3)), heuristic_repetition)

        return EvalResult(
            emotional_fit=float(raw.get("emotional_fit", 0.5)),
            persona_consistency=float(raw.get("persona_consistency", 0.5)),
            context_relevance=float(raw.get("context_relevance", 0.5)),
            memory_consistency=float(raw.get("memory_consistency", 0.9)),
            repetition=repetition,
        )

    @staticmethod
    def _heuristic_repetition(recent_assistant_turns: list[str], candidate: str) -> float:
        if not recent_assistant_turns:
            return 0.0
        cand_words = set(re.findall(r"[a-z']+", candidate.lower()))
        if not cand_words:
            return 0.0
        max_overlap = 0.0
        for prev in recent_assistant_turns[-4:]:
            prev_words = set(re.findall(r"[a-z']+", prev.lower()))
            if not prev_words:
                continue
            overlap = len(cand_words & prev_words) / len(cand_words | prev_words)
            max_overlap = max(max_overlap, overlap)
        return round(max_overlap, 3)
