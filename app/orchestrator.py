"""Ties the whole architecture together into one entry point: Persona.respond(...).

Pipeline (matches the "Normal conversation" flow in Section 12):
  User -> EQ Transformer -> Memory Retrieval -> Character State -> Dialogue Planner
       -> LLM -> Evaluator -> (accept | regenerate) -> Response -> Memory/State Update

Every generation attempt (not just the accepted one) is logged to the `candidates`
table, so training/export_dataset.py can build real chosen-vs-rejected DPO pairs from
actual system behavior rather than synthetic labels.

Production hardening:
- Observability: request metrics, regeneration tracking
- Graceful failure handling: LLM/retrieval failures don't corrupt state
- Prompt version tracking in candidates table
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from app.character import Character
from app.config import settings
from app.db import DB
from app.dialogue_planner import build_system_prompt, get_prompt_versions
from app.eq_estimator import EQState
from app.eq_estimator_trained import build_eq_estimator
from app.evaluator import EvalResult, ResponseEvaluator
from app.llm_client import LLMClient, build_llm_client
from app.memory import MemoryStore
from app.observability import (
    RequestMetrics, log_regeneration, metrics as obs_metrics, timed,
)
from app.scene_state import SceneStateStore
from app.user_state import UserStateStore

logger = logging.getLogger("persona_ai.orchestrator")


@dataclass
class TurnResult:
    response: str
    eq_state: EQState
    eval_result: EvalResult
    regenerations: int
    prompt_versions: dict | None = None


class Persona:
    def __init__(
        self,
        character: Character,
        db_path: str | None = None,
        database_url: str | None = None,
        llm: LLMClient | None = None,
    ):
        self.character = character
        # db_path kept for backward compat; database_url (or DATABASE_URL env) takes precedence
        resolved_url = database_url or (f"sqlite:///{db_path}" if db_path else None)
        self.db = DB(resolved_url)
        self.llm = llm or build_llm_client()
        self.eq_estimator = build_eq_estimator(self.llm)
        self.memory = MemoryStore(self.db)
        self.user_state = UserStateStore(self.db)
        self.scene_state = SceneStateStore(self.db)
        self.evaluator = ResponseEvaluator(self.llm)

    def _recent_turns(self, user_id: str, session_id: str | None, limit: int) -> list[dict]:
        with self.db.connect() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT role, content FROM turns WHERE user_id = :uid AND session_id = :sid "
                    "ORDER BY id DESC LIMIT :lim",
                    {"uid": user_id, "sid": session_id, "lim": limit},
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT role, content FROM turns WHERE user_id = :uid ORDER BY id DESC LIMIT :lim",
                    {"uid": user_id, "lim": limit},
                ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def _recent_assistant_texts(self, user_id: str, limit: int = 6) -> list[str]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT content FROM turns WHERE user_id = :uid AND role = 'assistant' "
                "ORDER BY id DESC LIMIT :lim",
                {"uid": user_id, "lim": limit},
            ).fetchall()
        return [r["content"] for r in reversed(rows)]

    def _session_turn_count(self, user_id: str, session_id: str) -> int:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM turns WHERE user_id = :uid AND session_id = :sid "
                "AND role = 'user'",
                {"uid": user_id, "sid": session_id},
            ).fetchone()
        return row["c"]

    def _store_turn(self, user_id: str, session_id: str, role: str, content: str, eq_state=None, eval_result=None):
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO turns (user_id, session_id, role, content, eq_state_json, "
                "eval_score_json, created_at) VALUES (:uid, :sid, :role, :content, :eq, :ev, :ts)",
                {
                    "uid": user_id,
                    "sid": session_id,
                    "role": role,
                    "content": content,
                    "eq": json.dumps(eq_state.to_dict()) if eq_state else None,
                    "ev": json.dumps(eval_result.to_dict()) if eval_result else None,
                    "ts": time.time(),
                },
            )

    def _store_candidate(
        self, user_id: str, session_id: str, user_message: str, system_prompt: str,
        response: str, eval_result: EvalResult, accepted: bool,
    ):
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO candidates (user_id, session_id, user_message, system_prompt, "
                "response, eval_json, accepted, created_at) VALUES "
                "(:uid, :sid, :msg, :sys, :resp, :ev, :acc, :ts)",
                {
                    "uid": user_id, "sid": session_id, "msg": user_message, "sys": system_prompt,
                    "resp": response, "ev": json.dumps(eval_result.to_dict()),
                    "acc": 1 if accepted else 0, "ts": time.time(),
                },
            )

    def respond(self, user_id: str, session_id: str, message: str) -> TurnResult:
        obs_metrics.increment("requests_total")
        request_start = time.perf_counter()

        self.db.ensure_user(user_id)

        # 1. Load prior state
        recent = self._recent_turns(user_id, session_id, settings.max_recent_turns)
        user_profile, relationship = self.user_state.load(user_id)
        prev_eq_row = self._latest_eq_state(user_id)
        prev_eq = EQState.from_dict(prev_eq_row) if prev_eq_row else EQState.initial()
        scene = self.scene_state.load(user_id)

        # 2. EQ Transformer: S_t = f(C_t, S_{t-1})
        with timed("eq_estimation_ms"):
            eq_state = self.eq_estimator.update(recent, message, prev_eq)

        # 3. Memory retrieval (user-isolated, configurable top-k)
        with timed("memory_retrieval_ms"):
            memories = self.memory.retrieve(user_id, message, k=settings.max_memories_retrieved)

        # 4. Generate -> Evaluate -> Regenerate loop (every attempt logged for DPO data)
        recent_assistant = self._recent_assistant_texts(user_id)
        feedback = None
        candidate = ""
        eval_result = None
        system = ""
        attempts = 0
        for attempt in range(settings.max_regeneration_attempts + 1):
            attempts = attempt + 1
            system = build_system_prompt(
                self.character, user_profile, relationship, eq_state, memories, scene, feedback
            )
            messages = recent + [{"role": "user", "content": message}]

            try:
                with timed("llm_generation_ms"):
                    candidate = self.llm.complete(system, messages)
            except Exception as exc:
                logger.error("LLM generation failed on attempt %d: %s", attempt + 1, exc)
                obs_metrics.increment("llm_generation_failures")
                # Use empty response rather than corrupting state
                candidate = ""
                break

            try:
                with timed("evaluation_ms"):
                    eval_result = self.evaluator.evaluate(self.character, eq_state, recent_assistant, candidate)
            except Exception as exc:
                logger.error("Evaluation failed on attempt %d: %s", attempt + 1, exc)
                # Accept the candidate without evaluation rather than losing it
                eval_result = EvalResult(
                    emotional_fit=0.5, persona_consistency=0.5, context_relevance=0.5,
                    memory_consistency=0.9, repetition=0.3,
                )
                break

            accepted = eval_result.overall >= settings.min_acceptable_score

            self._store_candidate(user_id, session_id, message, system, candidate, eval_result, accepted)

            if accepted:
                break
            feedback = self._feedback_from_eval(eval_result)

        # Track regenerations
        regenerations = attempts - 1
        log_regeneration(regenerations)

        # Default eval_result if none was produced
        if eval_result is None:
            eval_result = EvalResult(
                emotional_fit=0.5, persona_consistency=0.5, context_relevance=0.5,
                memory_consistency=0.9, repetition=0.3,
            )

        # 5. Persist turn + update state
        self._store_turn(user_id, session_id, "user", message, eq_state=eq_state)
        self._store_turn(user_id, session_id, "assistant", candidate, eval_result=eval_result)

        turn_count = self._session_turn_count(user_id, session_id)
        user_profile, relationship = self.user_state.apply_turn_update(
            user_id, user_profile, relationship, eq_state, turn_count
        )
        self.user_state.save(user_id, user_profile, relationship)

        # naive auto-memory: store high-intensity turns as episodic memories
        if eq_state.intensity > 0.6:
            self.memory.add(
                user_id, "episodic",
                f"User expressed {eq_state.intent} (intensity {eq_state.intensity:.2f}): \"{message[:200]}\"",
                importance=min(eq_state.intensity, 0.9),
            )

        # Record total request latency
        total_ms = (time.perf_counter() - request_start) * 1000
        obs_metrics.record_latency("request_total_ms", total_ms)

        return TurnResult(
            response=candidate,
            eq_state=eq_state,
            eval_result=eval_result,
            regenerations=regenerations,
            prompt_versions=get_prompt_versions(),
        )

    def _latest_eq_state(self, user_id: str) -> dict | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT eq_state_json FROM turns WHERE user_id = :uid AND eq_state_json IS NOT NULL "
                "ORDER BY id DESC LIMIT 1",
                {"uid": user_id},
            ).fetchone()
        if not row:
            return None
        return json.loads(row["eq_state_json"])

    @staticmethod
    def _feedback_from_eval(eval_result: EvalResult) -> str:
        issues = []
        if eval_result.emotional_fit < 0.5:
            issues.append("tone didn't match what the user needed")
        if eval_result.persona_consistency < 0.5:
            issues.append("didn't sound like the defined character")
        if eval_result.context_relevance < 0.5:
            issues.append("didn't actually engage with what the user said")
        if eval_result.repetition > 0.6:
            issues.append("too similar to a recent response — say something genuinely different")
        if eval_result.memory_consistency < 0.6:
            issues.append("referenced memories/history inconsistently")
        return "; ".join(issues) or "overall quality below threshold"
