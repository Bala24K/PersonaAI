"""Structured observability: logging, metrics, and tracing.

Provides structured logging and metrics collection for:
- Request latency
- LLM latency (per-provider)
- Retrieval latency
- Failures and retries
- Regeneration counts
- Token usage (when available from provider)
- Model/provider identification

NEVER logs secrets, API keys, or private user content.
"""
from __future__ import annotations

import logging
import time
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


logger = logging.getLogger("persona_ai.observability")


# ---------------------------------------------------------------------------
# Metric types
# ---------------------------------------------------------------------------

@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM API call."""
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    success: bool = True
    error_type: str = ""
    tokens_prompt: int = 0  # 0 = not available from provider
    tokens_completion: int = 0
    retries: int = 0
    is_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "success": self.success,
            "error_type": self.error_type,
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "retries": self.retries,
            "is_fallback": self.is_fallback,
        }


@dataclass
class RetrievalMetrics:
    """Metrics for a memory retrieval operation."""
    latency_ms: float = 0.0
    candidates_scanned: int = 0
    results_returned: int = 0
    user_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latency_ms": round(self.latency_ms, 2),
            "candidates_scanned": self.candidates_scanned,
            "results_returned": self.results_returned,
        }


@dataclass
class RequestMetrics:
    """Aggregate metrics for a single API request."""
    request_id: str = ""
    total_latency_ms: float = 0.0
    llm_calls: List[LLMCallMetrics] = field(default_factory=list)
    retrieval: Optional[RetrievalMetrics] = None
    regeneration_count: int = 0
    provider_used: str = ""
    model_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "llm_calls": [c.to_dict() for c in self.llm_calls],
            "retrieval": self.retrieval.to_dict() if self.retrieval else None,
            "regeneration_count": self.regeneration_count,
            "provider_used": self.provider_used,
            "model_used": self.model_used,
        }


# ---------------------------------------------------------------------------
# Metrics collector (thread-safe singleton)
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Thread-safe in-process metrics aggregation.

    This is deliberately simple — a production deployment would ship these
    to Prometheus/Datadog/CloudWatch rather than keeping them in-process.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._max_histogram_size = 10000  # bounded memory

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def record_latency(self, name: str, ms: float) -> None:
        with self._lock:
            hist = self._histograms[name]
            if len(hist) >= self._max_histogram_size:
                hist.pop(0)
            hist.append(ms)

    def get_counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    def get_latency_stats(self, name: str) -> Dict[str, float]:
        with self._lock:
            hist = self._histograms.get(name, [])
            if not hist:
                return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p99_ms": 0.0}
            sorted_h = sorted(hist)
            n = len(sorted_h)
            return {
                "count": n,
                "mean_ms": round(sum(sorted_h) / n, 2),
                "p50_ms": round(sorted_h[n // 2], 2),
                "p99_ms": round(sorted_h[min(int(n * 0.99), n - 1)], 2),
            }

    def snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of all current metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "latencies": {
                    name: self.get_latency_stats(name)
                    for name in self._histograms
                },
            }


# Global singleton
metrics = MetricsCollector()


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

@contextmanager
def timed(metric_name: str):
    """Context manager that records elapsed time to the metrics collector."""
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    metrics.record_latency(metric_name, elapsed_ms)


def log_llm_call(call_metrics: LLMCallMetrics) -> None:
    """Log an LLM call with structured fields. Never logs API keys or user content."""
    metrics.record_latency("llm_latency_ms", call_metrics.latency_ms)
    metrics.increment("llm_calls_total")
    if not call_metrics.success:
        metrics.increment("llm_calls_failed")
    if call_metrics.is_fallback:
        metrics.increment("llm_fallback_used")
    if call_metrics.retries > 0:
        metrics.increment("llm_retries_total", call_metrics.retries)
    if call_metrics.tokens_prompt > 0:
        metrics.increment("tokens_prompt_total", call_metrics.tokens_prompt)
    if call_metrics.tokens_completion > 0:
        metrics.increment("tokens_completion_total", call_metrics.tokens_completion)

    logger.info(
        "llm_call provider=%s model=%s latency_ms=%.2f success=%s retries=%d fallback=%s "
        "tokens_prompt=%d tokens_completion=%d",
        call_metrics.provider,
        call_metrics.model,
        call_metrics.latency_ms,
        call_metrics.success,
        call_metrics.retries,
        call_metrics.is_fallback,
        call_metrics.tokens_prompt,
        call_metrics.tokens_completion,
    )


def log_retrieval(retrieval_metrics: RetrievalMetrics) -> None:
    """Log a retrieval operation. Does NOT log user_id to avoid PII in logs."""
    metrics.record_latency("retrieval_latency_ms", retrieval_metrics.latency_ms)
    metrics.increment("retrieval_calls_total")

    logger.info(
        "retrieval latency_ms=%.2f candidates=%d returned=%d",
        retrieval_metrics.latency_ms,
        retrieval_metrics.candidates_scanned,
        retrieval_metrics.results_returned,
    )


def log_regeneration(count: int) -> None:
    """Log regeneration events."""
    if count > 0:
        metrics.increment("regenerations_total", count)
        logger.info("regeneration count=%d", count)


def log_validation_failure(schema_name: str) -> None:
    """Log a structured output validation failure."""
    metrics.increment(f"validation_failures_{schema_name}")
    metrics.increment("validation_failures_total")
    logger.warning("validation_failure schema=%s", schema_name)
