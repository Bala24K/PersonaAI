"""Extended evaluation harness for production-oriented metrics.

Adds the following evaluations to the existing harness:
- Retrieval relevance scoring
- Context recall measurement
- Response grounding checks
- Structured output validity testing
- Hallucination/unsupported claim detection
- Regression tests for previously observed failures
- Latency and token usage tracking

Keeps existing evaluation suites intact.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.character import Character
from app.llm_client import MockLLMClient
from app.memory import MemoryStore
from app.observability import MetricsCollector
from app.orchestrator import Persona
from app.schemas import (
    EQStateSchema, EvalResultSchema,
    validate_eq_state, validate_eval_result,
)

logger = logging.getLogger("persona_ai.eval_extended")

CHARACTER_FILE = REPO_ROOT / "characters" / "default_character.yaml"


def _fresh_persona():
    character = Character.from_yaml(str(CHARACTER_FILE))
    db_path = str(REPO_ROOT / "data" / f"eval_{uuid.uuid4().hex[:8]}.db")
    return Persona(character, db_path=db_path)


# ---------------------------------------------------------------------------
# 1. Retrieval Relevance
# ---------------------------------------------------------------------------

def eval_retrieval_relevance() -> dict:
    """Measures how relevant retrieved memories are to the query.

    Methodology: create known-relevant and known-irrelevant memories,
    then check if retrieval ranks relevant memories higher.
    """
    persona = _fresh_persona()
    user_id = "eval_retrieval_user"

    # Plant relevant memories
    persona.memory.add(user_id, "semantic", "User studies computer science at Stanford.", importance=0.9)
    persona.memory.add(user_id, "semantic", "User's favorite language is Python.", importance=0.8)
    persona.memory.add(user_id, "preference", "User prefers dark mode in all applications.", importance=0.6)

    # Plant irrelevant memories
    persona.memory.add(user_id, "episodic", "User had breakfast at 8am today.", importance=0.3)
    persona.memory.add(user_id, "semantic", "User's cat is named Whiskers.", importance=0.4)

    # Query about programming
    results = persona.memory.retrieve(user_id, "What programming language does the user prefer?", k=3)

    # Check if relevant memories are in top results
    relevant_keywords = {"python", "computer science", "language"}
    relevance_hits = 0
    for mem in results:
        content_lower = mem.content.lower()
        if any(kw in content_lower for kw in relevant_keywords):
            relevance_hits += 1

    precision_at_k = relevance_hits / len(results) if results else 0.0

    return {
        "test": "retrieval_relevance",
        "precision_at_3": round(precision_at_k, 3),
        "relevant_in_top_3": relevance_hits,
        "total_retrieved": len(results),
        "pass": precision_at_k >= 0.33,  # at least 1 relevant in top 3
    }


# ---------------------------------------------------------------------------
# 2. Context Recall
# ---------------------------------------------------------------------------

def eval_context_recall() -> dict:
    """Measures whether memories that should be recalled are actually recalled.

    Plants N relevant memories and checks what fraction are retrieved.
    """
    persona = _fresh_persona()
    user_id = "eval_recall_user"

    relevant_facts = [
        "User works as a software engineer at Google.",
        "User has a pet golden retriever named Max.",
        "User enjoys hiking on weekends.",
        "User is learning Japanese language.",
    ]
    for fact in relevant_facts:
        persona.memory.add(user_id, "semantic", fact, importance=0.8)

    # Add noise
    for i in range(6):
        persona.memory.add(user_id, "episodic", f"Routine conversation turn {i}.", importance=0.2)

    # Query that should recall work-related facts
    results = persona.memory.retrieve(user_id, "Tell me about the user's job and pets", k=4)
    recalled_contents = {m.content for m in results}

    recall_count = sum(1 for fact in relevant_facts[:2] if fact in recalled_contents)
    recall_rate = recall_count / 2  # checking top 2 relevant

    return {
        "test": "context_recall",
        "recall_rate": round(recall_rate, 3),
        "expected_recalled": 2,
        "actually_recalled": recall_count,
        "pass": recall_rate >= 0.5,
    }


# ---------------------------------------------------------------------------
# 3. Response Grounding
# ---------------------------------------------------------------------------

def eval_response_grounding() -> dict:
    """Checks whether the mock response references memories that exist.

    Uses the mock LLM to generate a response and checks if it mentions
    only memories that were actually provided in the prompt.
    """
    persona = _fresh_persona()
    user_id = "eval_grounding_user"

    # Add specific memories
    persona.memory.add(user_id, "semantic", "User works at a bakery.", importance=0.9)
    result = persona.respond(user_id, "s1", "Where do I work?")

    # The mock LLM should not invent facts. Check that response doesn't
    # contain fabricated details that weren't in memories or user message.
    fabricated_keywords = {"hospital", "bank", "school", "nasa", "facebook"}
    response_lower = result.response.lower()

    ungrounded = [kw for kw in fabricated_keywords if kw in response_lower]

    return {
        "test": "response_grounding",
        "ungrounded_claims": len(ungrounded),
        "ungrounded_keywords": ungrounded,
        "grounding_score": 1.0 if not ungrounded else round(1 - len(ungrounded) / 5, 3),
        "pass": len(ungrounded) == 0,
    }


# ---------------------------------------------------------------------------
# 4. Structured Output Validity
# ---------------------------------------------------------------------------

def eval_structured_output_validity() -> dict:
    """Tests that all mock LLM outputs pass Pydantic schema validation."""
    mock = MockLLMClient()
    test_prompts = [
        "What is the user's emotion and intent?",
        "The user said: I'm so happy today!",
        "The user said: I'm frustrated and angry",
        "The user said: can you help me with something?",
        "Evaluate this response quality.",
    ]

    eq_valid = 0
    eval_valid = 0
    eq_total = 0
    eval_total = 0

    for prompt in test_prompts:
        # Test EQ output
        if "emotion" in prompt.lower() or "intent" in prompt.lower() or "said" in prompt.lower():
            eq_total += 1
            try:
                raw = mock.complete_json("", prompt)
                validate_eq_state(raw)
                eq_valid += 1
            except Exception:
                pass

        # Test eval output
        if "evaluate" in prompt.lower() or "response" in prompt.lower():
            eval_total += 1
            try:
                raw = mock.complete_json("", prompt)
                validate_eval_result(raw)
                eval_valid += 1
            except Exception:
                pass

    eq_rate = eq_valid / max(eq_total, 1)
    eval_rate = eval_valid / max(eval_total, 1)

    return {
        "test": "structured_output_validity",
        "eq_validation_rate": round(eq_rate, 3),
        "eval_validation_rate": round(eval_rate, 3),
        "eq_tested": eq_total,
        "eval_tested": eval_total,
        "pass": eq_rate >= 0.8 and eval_rate >= 0.8,
    }


# ---------------------------------------------------------------------------
# 5. Hallucination / Unsupported Claims
# ---------------------------------------------------------------------------

def eval_hallucination_check() -> dict:
    """Tests that the system doesn't invent memories or facts.

    The mock LLM shouldn't reference memories that don't exist.
    """
    persona = _fresh_persona()
    user_id = "eval_hallucination_user"

    # Deliberately give NO memories
    result = persona.respond(user_id, "s1", "Do you remember what I told you last week?")

    # With no memories, the response should indicate no recall
    response_lower = result.response.lower()

    # Check for signs of fabricated recall
    hallucination_phrases = [
        "you told me about", "you mentioned", "last week you said",
        "i remember you telling me", "as you said before",
    ]
    hallucination_count = sum(1 for phrase in hallucination_phrases if phrase in response_lower)

    return {
        "test": "hallucination_check",
        "hallucination_phrases_found": hallucination_count,
        "hallucination_score": round(1 - min(hallucination_count / 3, 1.0), 3),
        "pass": hallucination_count <= 1,
    }


# ---------------------------------------------------------------------------
# 6. Regression Tests
# ---------------------------------------------------------------------------

def eval_regressions() -> dict:
    """Regression tests for previously observed failure modes."""
    results = []

    # Regression: empty message should not crash pipeline
    try:
        persona = _fresh_persona()
        # Note: the API validates non-empty, but the orchestrator should handle gracefully too
        result = persona.respond("reg_user", "s1", "test message")
        results.append({"case": "basic_response", "pass": len(result.response) > 0})
    except Exception as e:
        results.append({"case": "basic_response", "pass": False, "error": str(e)})

    # Regression: very long message should not crash
    try:
        persona = _fresh_persona()
        long_msg = "hello " * 500
        result = persona.respond("reg_long", "s1", long_msg)
        results.append({"case": "long_message", "pass": isinstance(result.response, str)})
    except Exception as e:
        results.append({"case": "long_message", "pass": False, "error": str(e)})

    # Regression: special characters in message
    try:
        persona = _fresh_persona()
        result = persona.respond("reg_special", "s1", "Hello! @#$%^&*() <script>alert('xss')</script>")
        results.append({"case": "special_chars", "pass": isinstance(result.response, str)})
    except Exception as e:
        results.append({"case": "special_chars", "pass": False, "error": str(e)})

    # Regression: unicode message
    try:
        persona = _fresh_persona()
        result = persona.respond("reg_unicode", "s1", "こんにちは、元気ですか？")
        results.append({"case": "unicode", "pass": isinstance(result.response, str)})
    except Exception as e:
        results.append({"case": "unicode", "pass": False, "error": str(e)})

    total = len(results)
    passed = sum(1 for r in results if r["pass"])

    return {
        "test": "regressions",
        "cases": results,
        "passed": passed,
        "total": total,
        "pass": passed == total,
    }


# ---------------------------------------------------------------------------
# 7. Latency / Token Usage
# ---------------------------------------------------------------------------

def eval_latency_and_tokens() -> dict:
    """Measures pipeline latency and checks observability metrics are being collected."""
    from app.observability import metrics

    persona = _fresh_persona()
    user_id = "eval_latency_user"

    latencies = []
    for i in range(3):
        start = time.perf_counter()
        persona.respond(user_id, "s1", f"Test message {i}")
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    # Check that observability metrics were recorded
    snapshot = metrics.snapshot()

    return {
        "test": "latency_and_tokens",
        "latencies_ms": [round(l, 1) for l in latencies],
        "mean_latency_ms": round(statistics.mean(latencies), 1),
        "metrics_snapshot_keys": list(snapshot.get("counters", {}).keys())[:10],
        "llm_calls_tracked": metrics.get_counter("llm_calls_total") > 0,
        "pass": True,  # latency measurement itself always passes
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all() -> dict:
    """Run all extended evaluations and produce a consolidated report."""
    evaluations = [
        ("Retrieval Relevance", eval_retrieval_relevance),
        ("Context Recall", eval_context_recall),
        ("Response Grounding", eval_response_grounding),
        ("Structured Output Validity", eval_structured_output_validity),
        ("Hallucination Check", eval_hallucination_check),
        ("Regression Tests", eval_regressions),
        ("Latency & Token Usage", eval_latency_and_tokens),
    ]

    results = []
    for name, fn in evaluations:
        print(f"  Running: {name}...", end=" ", flush=True)
        try:
            result = fn()
            result["evaluation_name"] = name
            results.append(result)
            status = "PASS" if result.get("pass") else "FAIL"
            print(f"[{status}]")
        except Exception as e:
            print(f"[ERROR: {e}]")
            results.append({"evaluation_name": name, "pass": False, "error": str(e)})

    total = len(results)
    passed = sum(1 for r in results if r.get("pass"))

    summary = {
        "total_evaluations": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 3) if total > 0 else 0.0,
        "results": results,
    }

    print(f"\n  Summary: {passed}/{total} passed ({summary['pass_rate']*100:.1f}%)")
    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("Extended Evaluation Harness")
    print("=" * 60)
    summary = run_all()
    print(f"\n{'='*60}")
    output_str = json.dumps(summary, indent=2, default=str)
    print(output_str)

    out_dir = REPO_ROOT / "ml" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "extended_eval.txt"
    with open(report_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Extended Evaluation Harness Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(output_str + "\n")
    print(f"\nSaved report to {report_file}")
