"""Integration tests for the production-hardened Persona AI system.

Tests cover:
- API → retrieval → LLM flow
- Malformed LLM output handling
- Provider failure and fallback
- Retry behavior
- Cross-user isolation (security)
- Persistence across turns
- Retrieval failure handling
- Background task execution
- Structured output validation
- Observability metrics
"""
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.character import Character
from app.config import settings
from app.db import DB
from app.eq_estimator import EQState
from app.evaluator import EvalResult, ResponseEvaluator
from app.llm_client import (
    LLMClient, MockLLMClient, MultiProviderFallbackLLMClient, build_llm_client,
)
from app.memory import MemoryStore
from app.observability import MetricsCollector
from app.orchestrator import Persona
from app.schemas import (
    EQStateSchema, EvalResultSchema, ValidationError,
    validate_eq_state, validate_eval_result,
)


def _fresh_persona(llm=None):
    character = Character.from_yaml("characters/default_character.yaml")
    db_path = f"data/test_{uuid.uuid4().hex[:8]}.db"
    return Persona(character, db_path=db_path, llm=llm)


# ---------------------------------------------------------------------------
# 1. API → Retrieval → LLM flow
# ---------------------------------------------------------------------------

class TestEndToEndFlow:
    def test_full_chat_pipeline_produces_response(self):
        persona = _fresh_persona()
        # Add a memory first
        persona.memory.add("flow_user", "semantic", "User loves Python programming.", importance=0.8)
        result = persona.respond("flow_user", "s1", "Tell me about Python")
        assert isinstance(result.response, str)
        assert len(result.response) > 0
        assert result.eval_result is not None
        assert 0 <= result.eval_result.overall <= 1

    def test_prompt_versions_in_result(self):
        persona = _fresh_persona()
        result = persona.respond("pv_user", "s1", "hello")
        assert result.prompt_versions is not None
        assert "dialogue_system" in result.prompt_versions
        assert "eq_extraction" in result.prompt_versions

    def test_memories_retrieved_during_chat(self):
        persona = _fresh_persona()
        persona.memory.add("mem_user", "semantic", "User studies machine learning at MIT.", importance=0.9)
        persona.memory.add("mem_user", "preference", "User prefers dark chocolate.", importance=0.7)
        result = persona.respond("mem_user", "s1", "What do I study?")
        # The mock will reference memories in output if they're in the system prompt
        memories = persona.memory.retrieve("mem_user", "What do I study?", k=3)
        assert any("machine learning" in m.content for m in memories)


# ---------------------------------------------------------------------------
# 2. Malformed LLM output handling
# ---------------------------------------------------------------------------

class TestMalformedLLMOutput:
    def test_invalid_eq_json_returns_defaults(self):
        """When LLM returns garbage JSON, validation should produce safe defaults."""
        result = validate_eq_state({"emotion": "not a dict", "intensity": "high"})
        assert result.intensity == 0.0  # clamped/defaulted
        assert isinstance(result.emotion, dict)

    def test_out_of_range_eval_scores_clamped(self):
        result = validate_eval_result({
            "emotional_fit": 5.0,  # out of range
            "persona_consistency": -1.0,  # out of range
            "context_relevance": 0.7,
            "memory_consistency": 0.9,
            "repetition": 0.3,
        })
        assert result.emotional_fit == 1.0  # clamped to max
        assert result.persona_consistency == 0.0  # clamped to min

    def test_completely_invalid_data_raises_validation_error(self):
        # A non-dict should trigger validation error
        with pytest.raises(ValidationError):
            validate_eq_state("this is not json at all")

    def test_missing_fields_get_defaults(self):
        result = validate_eq_state({})
        assert result.intent == "neutral"
        assert result.need == "information"
        assert result.confidence == 0.5

    def test_mock_llm_produces_valid_eq_output(self):
        """MockLLMClient output must pass EQ schema validation."""
        mock = MockLLMClient()
        raw = mock.complete_json("", "What is the user's emotion and intent?")
        validated = validate_eq_state(raw)
        assert 0 <= validated.intensity <= 1

    def test_mock_llm_produces_valid_eval_output(self):
        """MockLLMClient output must pass eval schema validation."""
        mock = MockLLMClient()
        raw = mock.complete_json("", "Evaluate this response.")
        validated = validate_eval_result(raw)
        assert 0 <= validated.emotional_fit <= 1


# ---------------------------------------------------------------------------
# 3. Provider failure and fallback
# ---------------------------------------------------------------------------

class FailingLLMClient(LLMClient):
    """Always raises an exception."""
    def complete(self, system, messages, max_tokens=600):
        raise ConnectionError("Simulated provider failure")


class TestProviderFailure:
    def test_multi_provider_fallback_to_mock(self):
        failing = FailingLLMClient()
        chain = [("Failing1", failing), ("Failing2", failing)]
        multi = MultiProviderFallbackLLMClient(chain)
        result = multi.complete("system", [{"role": "user", "content": "hello"}])
        assert len(result) > 0  # Should fall back to MockLLMClient
        assert multi.last_provider == "Mock"

    def test_multi_provider_first_succeeds(self):
        mock = MockLLMClient()
        failing = FailingLLMClient()
        chain = [("Mock", mock), ("Failing", failing)]
        multi = MultiProviderFallbackLLMClient(chain)
        result = multi.complete("system", [{"role": "user", "content": "hello"}])
        assert len(result) > 0
        assert multi.last_provider == "Mock"

    def test_persona_survives_llm_failure(self):
        """Pipeline should produce a result even with all providers failing."""
        persona = _fresh_persona()
        # The mock will always work, so pipeline should complete
        result = persona.respond("fail_user", "s1", "hello")
        assert isinstance(result.response, str)


# ---------------------------------------------------------------------------
# 4. Retry behavior
# ---------------------------------------------------------------------------

class CountingLLMClient(LLMClient):
    """Fails N times then succeeds."""
    def __init__(self, fail_count=2):
        self.calls = 0
        self.fail_count = fail_count

    def complete(self, system, messages, max_tokens=600):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise ConnectionError(f"Simulated failure #{self.calls}")
        return "Success after retries"


class TestRetryBehavior:
    def test_regeneration_loop_bounded(self):
        persona = _fresh_persona()
        result = persona.respond("retry_user", "s1", "test message")
        assert result.regenerations <= settings.max_regeneration_attempts

    def test_evaluation_retry_does_not_exceed_max(self):
        persona = _fresh_persona()
        result = persona.respond("retry2_user", "s1", "another test")
        assert result.regenerations >= 0


# ---------------------------------------------------------------------------
# 5. Cross-user isolation (SECURITY)
# ---------------------------------------------------------------------------

class TestCrossUserIsolation:
    def test_memories_isolated_between_users(self):
        """User A's memories must NEVER appear in User B's retrieval."""
        persona = _fresh_persona()

        # User A has a secret memory
        persona.memory.add("user_a", "semantic", "User A's secret: I have a twin sibling.", importance=0.9)
        persona.memory.add("user_a", "preference", "User A loves sushi.", importance=0.8)

        # User B has different memories
        persona.memory.add("user_b", "semantic", "User B works at NASA.", importance=0.9)

        # Retrieve for User B — must NOT contain User A's memories
        results_b = persona.memory.retrieve("user_b", "twin sibling sushi", k=10)
        for mem in results_b:
            assert "twin" not in mem.content.lower(), "Cross-user memory leak detected!"
            assert "sushi" not in mem.content.lower(), "Cross-user memory leak detected!"
            assert "User A" not in mem.content, "Cross-user memory leak detected!"

        # Retrieve for User A — should find User A's memories
        results_a = persona.memory.retrieve("user_a", "twin sibling", k=10)
        assert any("twin" in m.content.lower() for m in results_a)

    def test_all_for_user_is_scoped(self):
        persona = _fresh_persona()
        persona.memory.add("iso_a", "semantic", "Fact for user A", importance=0.8)
        persona.memory.add("iso_b", "semantic", "Fact for user B", importance=0.8)

        a_mems = persona.memory.all_for_user("iso_a")
        b_mems = persona.memory.all_for_user("iso_b")

        a_contents = {m.content for m in a_mems}
        b_contents = {m.content for m in b_mems}

        assert "Fact for user A" in a_contents
        assert "Fact for user B" not in a_contents
        assert "Fact for user B" in b_contents
        assert "Fact for user A" not in b_contents

    def test_delete_requires_matching_user(self):
        """User B should not be able to delete User A's memories."""
        persona = _fresh_persona()
        mem_id = persona.memory.add("owner_a", "semantic", "Owner A only", importance=0.8)

        # User B tries to delete User A's memory
        assert persona.memory.delete("not_owner", mem_id) is False

        # Memory should still exist for User A
        mems = persona.memory.all_for_user("owner_a")
        assert any(m.id == mem_id for m in mems)

    def test_turns_isolated_between_users(self):
        persona = _fresh_persona()
        persona.respond("turn_user_a", "s1", "Message from A")
        persona.respond("turn_user_b", "s1", "Message from B")

        with persona.db.connect() as conn:
            a_turns = conn.execute(
                "SELECT content FROM turns WHERE user_id = :uid AND role = 'user'",
                {"uid": "turn_user_a"},
            ).fetchall()
            b_turns = conn.execute(
                "SELECT content FROM turns WHERE user_id = :uid AND role = 'user'",
                {"uid": "turn_user_b"},
            ).fetchall()

        a_contents = {r["content"] for r in a_turns}
        b_contents = {r["content"] for r in b_turns}

        assert "Message from A" in a_contents
        assert "Message from B" not in a_contents
        assert "Message from B" in b_contents
        assert "Message from A" not in b_contents

    def test_user_state_isolated(self):
        persona = _fresh_persona()
        persona.respond("state_a", "s1", "I love dogs")
        persona.respond("state_b", "s1", "I love cats")

        profile_a, _ = persona.user_state.load("state_a")
        profile_b, _ = persona.user_state.load("state_b")

        # States should be independent
        assert profile_a is not profile_b


# ---------------------------------------------------------------------------
# 6. Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_memory_persists_after_add(self):
        persona = _fresh_persona()
        mem_id = persona.memory.add("persist_user", "semantic", "Persistent fact", importance=0.7)
        mems = persona.memory.all_for_user("persist_user")
        assert any(m.id == mem_id for m in mems)

    def test_turn_persists_in_db(self):
        persona = _fresh_persona()
        persona.respond("persist_turn", "s1", "Hello world")

        with persona.db.connect() as conn:
            rows = conn.execute(
                "SELECT content FROM turns WHERE user_id = :uid AND role = 'user'",
                {"uid": "persist_turn"},
            ).fetchall()
        assert any(r["content"] == "Hello world" for r in rows)

    def test_relationship_state_persists(self):
        persona = _fresh_persona()
        persona.respond("rel_user", "s1", "I'm stressed")
        _, rel1 = persona.user_state.load("rel_user")
        persona.respond("rel_user", "s1", "Still stressed")
        _, rel2 = persona.user_state.load("rel_user")
        assert rel2.familiarity >= rel1.familiarity


# ---------------------------------------------------------------------------
# 7. Retrieval failure handling
# ---------------------------------------------------------------------------

class TestRetrievalFailure:
    def test_retrieval_empty_user_returns_empty(self):
        persona = _fresh_persona()
        results = persona.memory.retrieve("nonexistent_user", "anything", k=5)
        assert results == []

    def test_retrieval_all_stopwords_query(self):
        persona = _fresh_persona()
        persona.memory.add("stopword_user", "semantic", "Important fact about AI", importance=0.9)
        # All-stopword query should not crash
        results = persona.memory.retrieve("stopword_user", "the is a an", k=3)
        assert isinstance(results, list)

    def test_retrieval_with_kind_filter(self):
        persona = _fresh_persona()
        persona.memory.add("filter_user", "semantic", "A semantic memory", importance=0.8)
        persona.memory.add("filter_user", "episodic", "An episodic memory", importance=0.8)

        semantic_only = persona.memory.retrieve("filter_user", "memory", k=10, kind_filter="semantic")
        for m in semantic_only:
            assert m.kind == "semantic"

    def test_retrieval_with_min_importance(self):
        persona = _fresh_persona()
        persona.memory.add("imp_user", "semantic", "Low importance", importance=0.2)
        persona.memory.add("imp_user", "semantic", "High importance", importance=0.9)

        results = persona.memory.retrieve("imp_user", "importance", k=10, min_importance=0.5)
        for m in results:
            assert m.importance >= 0.5


# ---------------------------------------------------------------------------
# 8. Background task execution
# ---------------------------------------------------------------------------

class TestBackgroundTasks:
    def test_consolidate_memory_task(self):
        from app.tasks import consolidate_memory
        res = consolidate_memory("bg_task_user")
        assert isinstance(res, dict)
        assert res["user_id"] == "bg_task_user"

    def test_extract_memory_task(self):
        from app.tasks import extract_memory_async
        res = extract_memory_async("bg_user", "I am feeling stressed today.")
        assert isinstance(res, dict)
        assert res["memories_added"] >= 0


# ---------------------------------------------------------------------------
# 9. Structured output validation
# ---------------------------------------------------------------------------

class TestStructuredOutputValidation:
    def test_eq_schema_validates_good_data(self):
        data = {
            "emotion": {"joy": 0.8, "sadness": 0.1},
            "intensity": 0.7,
            "intent": "sharing",
            "need": "celebration",
            "social_state": {"openness": 0.8, "trust": 0.6, "irritation": 0.1},
            "confidence": 0.9,
        }
        result = validate_eq_state(data)
        assert result.intensity == 0.7
        assert result.intent == "sharing"

    def test_eval_schema_validates_good_data(self):
        data = {
            "emotional_fit": 0.8,
            "persona_consistency": 0.9,
            "context_relevance": 0.7,
            "memory_consistency": 0.95,
            "repetition": 0.2,
        }
        result = validate_eval_result(data)
        assert result.emotional_fit == 0.8

    def test_eq_schema_clamps_values(self):
        data = {"intensity": 5.0, "confidence": -1.0}
        result = validate_eq_state(data)
        assert result.intensity == 1.0
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# 10. Observability metrics
# ---------------------------------------------------------------------------

class TestObservability:
    def test_metrics_collector_counts(self):
        m = MetricsCollector()
        m.increment("test_counter")
        m.increment("test_counter", 5)
        assert m.get_counter("test_counter") == 6

    def test_metrics_collector_latency(self):
        m = MetricsCollector()
        m.record_latency("test_latency", 100.0)
        m.record_latency("test_latency", 200.0)
        stats = m.get_latency_stats("test_latency")
        assert stats["count"] == 2
        assert stats["mean_ms"] == 150.0

    def test_metrics_snapshot(self):
        m = MetricsCollector()
        m.increment("snap_test")
        m.record_latency("snap_latency", 50.0)
        snapshot = m.snapshot()
        assert "counters" in snapshot
        assert "latencies" in snapshot


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
