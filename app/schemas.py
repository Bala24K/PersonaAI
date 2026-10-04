"""Pydantic v2 schemas for structured LLM output validation.

Every JSON response from an LLM that enters application state or database must be
validated through one of these schemas first. Raw LLM output is never trusted as
application state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# EQ State (emotional/social state extraction from LLM)
# ---------------------------------------------------------------------------

class EQStateSchema(BaseModel):
    """Validates the structured EQ state JSON returned by the LLM or trained model."""

    emotion: Dict[str, float] = Field(
        default_factory=dict,
        description="Dominant emotions as label->intensity (0-1) mapping",
    )
    intensity: float = Field(default=0.0)
    intent: str = Field(default="neutral")
    need: str = Field(default="information")
    social_state: Dict[str, float] = Field(
        default_factory=lambda: {"openness": 0.5, "trust": 0.5, "irritation": 0.0},
    )
    confidence: float = Field(default=0.5)

    @field_validator("emotion", mode="before")
    @classmethod
    def clamp_emotion_values(cls, v: Any) -> Dict[str, float]:
        if not isinstance(v, dict):
            return {}
        return {str(k): max(0.0, min(1.0, float(val))) for k, val in v.items()}

    @field_validator("social_state", mode="before")
    @classmethod
    def normalize_social_state(cls, v: Any) -> Dict[str, float]:
        if not isinstance(v, dict):
            return {"openness": 0.5, "trust": 0.5, "irritation": 0.0}
        result = {}
        for key in ("openness", "trust", "irritation"):
            raw = v.get(key, 0.5 if key != "irritation" else 0.0)
            result[key] = max(0.0, min(1.0, float(raw)))
        return result

    @field_validator("intensity", "confidence", mode="before")
    @classmethod
    def clamp_unit_floats(cls, v: Any) -> float:
        try:
            val = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, val))

    @field_validator("intent", mode="before")
    @classmethod
    def sanitize_intent(cls, v: Any) -> str:
        return str(v).strip()[:50] if v else "neutral"

    @field_validator("need", mode="before")
    @classmethod
    def sanitize_need(cls, v: Any) -> str:
        return str(v).strip()[:50] if v else "information"


# ---------------------------------------------------------------------------
# Evaluation Result (response quality scores from LLM critic)
# ---------------------------------------------------------------------------

class EvalResultSchema(BaseModel):
    """Validates the evaluation scores returned by the LLM self-critique."""

    emotional_fit: float = Field(default=0.5, ge=0.0, le=1.0)
    persona_consistency: float = Field(default=0.5, ge=0.0, le=1.0)
    context_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    memory_consistency: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition: float = Field(default=0.3, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def coerce_floats(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for key in ("emotional_fit", "persona_consistency", "context_relevance",
                     "memory_consistency", "repetition"):
            if key in data:
                try:
                    data[key] = max(0.0, min(1.0, float(data[key])))
                except (TypeError, ValueError):
                    data[key] = 0.5
        return data


# ---------------------------------------------------------------------------
# Memory extraction result (structured output from LLM memory analysis)
# ---------------------------------------------------------------------------

class ExtractedMemory(BaseModel):
    """A single memory item extracted by the LLM from user text."""
    kind: str = Field(description="One of: semantic, episodic, preference, relational")
    content: str = Field(description="The extracted fact or observation")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("kind", mode="before")
    @classmethod
    def validate_kind(cls, v: Any) -> str:
        valid = {"semantic", "episodic", "preference", "relational"}
        s = str(v).strip().lower()
        return s if s in valid else "semantic"


class MemoryExtractionResult(BaseModel):
    """Validates batch memory extraction output from LLM."""
    memories: List[ExtractedMemory] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval relevance scoring (for evaluation harness)
# ---------------------------------------------------------------------------

class RetrievalRelevanceScore(BaseModel):
    """Scores how relevant retrieved memories are to a query."""
    query: str
    retrieved_ids: List[int] = Field(default_factory=list)
    relevance_scores: List[float] = Field(default_factory=list)
    mean_relevance: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Response grounding check (for evaluation harness)
# ---------------------------------------------------------------------------

class GroundingCheckResult(BaseModel):
    """Checks whether response claims are grounded in provided context."""
    grounded_claims: int = Field(default=0, ge=0)
    ungrounded_claims: int = Field(default=0, ge=0)
    grounding_score: float = Field(default=1.0, ge=0.0, le=1.0)
    unsupported_statements: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when structured LLM output fails validation."""

    def __init__(self, schema_name: str, errors: Any, raw_data: Any = None):
        self.schema_name = schema_name
        self.errors = errors
        self.raw_data = raw_data
        super().__init__(f"Validation failed for {schema_name}: {errors}")


def validate_eq_state(raw: dict) -> EQStateSchema:
    """Validate raw LLM output as EQ state, raising ValidationError on failure."""
    try:
        return EQStateSchema.model_validate(raw)
    except Exception as exc:
        raise ValidationError("EQStateSchema", str(exc), raw) from exc


def validate_eval_result(raw: dict) -> EvalResultSchema:
    """Validate raw LLM output as eval result, raising ValidationError on failure."""
    try:
        return EvalResultSchema.model_validate(raw)
    except Exception as exc:
        raise ValidationError("EvalResultSchema", str(exc), raw) from exc
