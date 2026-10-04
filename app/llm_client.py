"""Main LLM Generator (Section 9 of pitch doc), pluggable and multi-provider resilient so
the system gracefully falls back across providers (OpenRouter, Groq, Cohere, Gemini, Anthropic)
down to the deterministic mock responder if external services encounter outages or rate limits.

Production hardening:
- Configurable timeout per provider
- Bounded retry with exponential backoff
- Provider fallback chain
- Structured output validation via Pydantic schemas
- Observability metrics for every call
- Failures degrade gracefully — never corrupt persistent state
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.config import settings
from app.observability import LLMCallMetrics, log_llm_call, log_validation_failure
from app.schemas import EQStateSchema, EvalResultSchema, ValidationError

logger = logging.getLogger("persona_ai.llm")

# Default timeout and retry settings
DEFAULT_TIMEOUT_S = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_BASE = 1.5


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, messages: list[dict], max_tokens: int = 600) -> str:
        """messages: list of {"role": "user"|"assistant", "content": str}"""
        raise NotImplementedError

    def complete_json(self, system: str, user_prompt: str, max_tokens: int = 500) -> dict:
        """Ask for structured JSON output only. Used by EQ estimator and evaluator."""
        full_system = system + "\n\nRespond with ONLY a valid JSON object. No prose, no markdown fences."
        raw = self.complete(full_system, [{"role": "user", "content": user_prompt}], max_tokens)
        return _extract_json(raw)

    def complete_json_validated(
        self,
        system: str,
        user_prompt: str,
        schema_name: str,
        max_tokens: int = 500,
        max_retries: int = 2,
    ) -> dict:
        """Ask for structured JSON and validate against a Pydantic schema.

        Retries up to max_retries times if output fails validation.
        Falls back to schema defaults if all attempts fail.
        """
        from app.schemas import validate_eq_state, validate_eval_result

        validators = {
            "EQStateSchema": validate_eq_state,
            "EvalResultSchema": validate_eval_result,
        }
        validator = validators.get(schema_name)
        if not validator:
            # No validator registered — fall back to raw JSON extraction
            return self.complete_json(system, user_prompt, max_tokens)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                raw = self.complete_json(system, user_prompt, max_tokens)
                validated = validator(raw)
                return validated.model_dump()
            except ValidationError as exc:
                last_error = exc
                log_validation_failure(schema_name)
                logger.warning(
                    "Structured output validation failed (attempt %d/%d) schema=%s: %s",
                    attempt + 1, max_retries + 1, schema_name, exc,
                )
                if attempt < max_retries:
                    # Add validation feedback to the prompt for retry
                    user_prompt = (
                        user_prompt
                        + f"\n\n[VALIDATION ERROR: your previous response failed schema validation: {exc}. "
                        f"Please output valid JSON matching the required schema exactly.]"
                    )
                continue
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "JSON extraction failed (attempt %d/%d): %s",
                    attempt + 1, max_retries + 1, exc,
                )
                continue

        # All attempts failed — return schema defaults rather than corrupting state
        logger.error(
            "All %d validation attempts failed for %s, returning schema defaults. Last error: %s",
            max_retries + 1, schema_name, last_error,
        )
        if schema_name == "EQStateSchema":
            return EQStateSchema().model_dump()
        elif schema_name == "EvalResultSchema":
            return EvalResultSchema().model_dump()
        return {}


class OpenRouterLLMClient(LLMClient):
    """Calls OpenRouter chat completions API (OpenAI-compatible)."""

    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.3-70b-instruct",
                 timeout: float = DEFAULT_TIMEOUT_S, max_retries: int = DEFAULT_MAX_RETRIES):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

    def complete(self, system: str, messages: list[dict], max_tokens: int = 600) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        api_messages = [{"role": "system", "content": system}]
        for m in messages:
            api_messages.append({"role": m.get("role", "user"), "content": str(m.get("content", ""))})

        payload = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        req_data = json.dumps(payload).encode("utf-8")

        call_metrics = LLMCallMetrics(provider="OpenRouter", model=self._model)
        start = time.perf_counter()

        for attempt in range(self._max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    data=req_data,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://persona-ai.local",
                        "X-Title": "PersonaAI",
                    },
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    result = data["choices"][0]["message"]["content"].strip()
                    # Extract token usage if available
                    usage = data.get("usage", {})
                    call_metrics.tokens_prompt = usage.get("prompt_tokens", 0)
                    call_metrics.tokens_completion = usage.get("completion_tokens", 0)
                    call_metrics.latency_ms = (time.perf_counter() - start) * 1000
                    call_metrics.retries = attempt
                    log_llm_call(call_metrics)
                    return result
            except urllib.error.HTTPError as err:
                if err.code in (429, 500, 502, 503, 504) and attempt < self._max_retries - 1:
                    call_metrics.retries = attempt + 1
                    time.sleep(DEFAULT_RETRY_BACKOFF_BASE * (attempt + 1))
                    continue
                call_metrics.success = False
                call_metrics.error_type = f"HTTP_{err.code}"
                call_metrics.latency_ms = (time.perf_counter() - start) * 1000
                log_llm_call(call_metrics)
                raise
            except Exception as exc:
                if attempt < self._max_retries - 1:
                    call_metrics.retries = attempt + 1
                    time.sleep(DEFAULT_RETRY_BACKOFF_BASE * (attempt + 1))
                    continue
                call_metrics.success = False
                call_metrics.error_type = type(exc).__name__
                call_metrics.latency_ms = (time.perf_counter() - start) * 1000
                log_llm_call(call_metrics)
                raise

        call_metrics.success = False
        call_metrics.error_type = "max_retries_exceeded"
        call_metrics.latency_ms = (time.perf_counter() - start) * 1000
        log_llm_call(call_metrics)
        raise RuntimeError("OpenRouter: max retries exceeded")


class GroqLLMClient(LLMClient):
    """Calls Groq Cloud API (OpenAI-compatible, ultra low-latency)."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile",
                 timeout: float = DEFAULT_TIMEOUT_S, max_retries: int = DEFAULT_MAX_RETRIES):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

    def complete(self, system: str, messages: list[dict], max_tokens: int = 600) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        api_messages = [{"role": "system", "content": system}]
        for m in messages:
            api_messages.append({"role": m.get("role", "user"), "content": str(m.get("content", ""))})

        payload = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        req_data = json.dumps(payload).encode("utf-8")

        call_metrics = LLMCallMetrics(provider="Groq", model=self._model)
        start = time.perf_counter()

        for attempt in range(self._max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    data=req_data,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    result = data["choices"][0]["message"]["content"].strip()
                    usage = data.get("usage", {})
                    call_metrics.tokens_prompt = usage.get("prompt_tokens", 0)
                    call_metrics.tokens_completion = usage.get("completion_tokens", 0)
                    call_metrics.latency_ms = (time.perf_counter() - start) * 1000
                    call_metrics.retries = attempt
                    log_llm_call(call_metrics)
                    return result
            except urllib.error.HTTPError as err:
                if err.code in (429, 500, 502, 503, 504) and attempt < self._max_retries - 1:
                    call_metrics.retries = attempt + 1
                    time.sleep(DEFAULT_RETRY_BACKOFF_BASE * (attempt + 1))
                    continue
                call_metrics.success = False
                call_metrics.error_type = f"HTTP_{err.code}"
                call_metrics.latency_ms = (time.perf_counter() - start) * 1000
                log_llm_call(call_metrics)
                raise
            except Exception as exc:
                if attempt < self._max_retries - 1:
                    call_metrics.retries = attempt + 1
                    time.sleep(DEFAULT_RETRY_BACKOFF_BASE * (attempt + 1))
                    continue
                call_metrics.success = False
                call_metrics.error_type = type(exc).__name__
                call_metrics.latency_ms = (time.perf_counter() - start) * 1000
                log_llm_call(call_metrics)
                raise

        call_metrics.success = False
        call_metrics.error_type = "max_retries_exceeded"
        call_metrics.latency_ms = (time.perf_counter() - start) * 1000
        log_llm_call(call_metrics)
        raise RuntimeError("Groq: max retries exceeded")


class CohereLLMClient(LLMClient):
    """Calls Cohere v2 Chat API."""

    def __init__(self, api_key: str, model: str = "command-r-plus",
                 timeout: float = DEFAULT_TIMEOUT_S, max_retries: int = DEFAULT_MAX_RETRIES):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

    def complete(self, system: str, messages: list[dict], max_tokens: int = 600) -> str:
        url = "https://api.cohere.com/v2/chat"
        api_messages = [{"role": "system", "content": system}]
        for m in messages:
            role = "assistant" if m.get("role") == "assistant" else "user"
            api_messages.append({"role": role, "content": {"type": "text", "text": str(m.get("content", ""))}})

        payload = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        req_data = json.dumps(payload).encode("utf-8")

        call_metrics = LLMCallMetrics(provider="Cohere", model=self._model)
        start = time.perf_counter()

        for attempt in range(self._max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    data=req_data,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    result = data["message"]["content"][0]["text"].strip()
                    call_metrics.latency_ms = (time.perf_counter() - start) * 1000
                    call_metrics.retries = attempt
                    log_llm_call(call_metrics)
                    return result
            except urllib.error.HTTPError as err:
                if err.code in (429, 500, 502, 503, 504) and attempt < self._max_retries - 1:
                    call_metrics.retries = attempt + 1
                    time.sleep(DEFAULT_RETRY_BACKOFF_BASE * (attempt + 1))
                    continue
                call_metrics.success = False
                call_metrics.error_type = f"HTTP_{err.code}"
                call_metrics.latency_ms = (time.perf_counter() - start) * 1000
                log_llm_call(call_metrics)
                raise
            except Exception as exc:
                if attempt < self._max_retries - 1:
                    call_metrics.retries = attempt + 1
                    time.sleep(DEFAULT_RETRY_BACKOFF_BASE * (attempt + 1))
                    continue
                call_metrics.success = False
                call_metrics.error_type = type(exc).__name__
                call_metrics.latency_ms = (time.perf_counter() - start) * 1000
                log_llm_call(call_metrics)
                raise

        call_metrics.success = False
        call_metrics.error_type = "max_retries_exceeded"
        call_metrics.latency_ms = (time.perf_counter() - start) * 1000
        log_llm_call(call_metrics)
        raise RuntimeError("Cohere: max retries exceeded")


class GeminiLLMClient(LLMClient):
    """Calls Google Gemini REST API using standard urllib with exponential retries."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash",
                 timeout: float = DEFAULT_TIMEOUT_S, max_retries: int = DEFAULT_MAX_RETRIES):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

    def complete(self, system: str, messages: list[dict], max_tokens: int = 600) -> str:
        gemini_contents = []
        for msg in messages:
            role = "model" if msg.get("role") == "assistant" else "user"
            content = msg.get("content", "")
            gemini_contents.append({"role": role, "parts": [{"text": str(content)}]})

        payload = {
            "system_instruction": {
                "parts": [{"text": system}]
            },
            "contents": gemini_contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7,
            }
        }
        req_data = json.dumps(payload).encode("utf-8")

        candidate_models = [self._model]
        for fallback in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        call_metrics = LLMCallMetrics(provider="Gemini", model=self._model)
        start = time.perf_counter()

        for current_model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={self._api_key}"
            req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})

            for attempt in range(self._max_retries):
                try:
                    with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        candidates = data.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"].get("parts", [])
                            text_blocks = [p.get("text", "") for p in parts if "text" in p]
                            result = "\n".join(text_blocks).strip()
                            # Extract token usage if available
                            usage = data.get("usageMetadata", {})
                            call_metrics.tokens_prompt = usage.get("promptTokenCount", 0)
                            call_metrics.tokens_completion = usage.get("candidatesTokenCount", 0)
                            call_metrics.model = current_model
                            call_metrics.latency_ms = (time.perf_counter() - start) * 1000
                            call_metrics.retries = attempt
                            if current_model != self._model:
                                call_metrics.is_fallback = True
                            log_llm_call(call_metrics)
                            return result
                        return ""
                except urllib.error.HTTPError as err:
                    if err.code == 404:
                        break
                    if err.code in (429, 500, 502, 503, 504) and attempt < self._max_retries - 1:
                        time.sleep(DEFAULT_RETRY_BACKOFF_BASE * (attempt + 1))
                        continue
                    call_metrics.success = False
                    call_metrics.error_type = f"HTTP_{err.code}"
                    call_metrics.latency_ms = (time.perf_counter() - start) * 1000
                    log_llm_call(call_metrics)
                    raise
                except Exception:
                    if attempt < self._max_retries - 1:
                        time.sleep(DEFAULT_RETRY_BACKOFF_BASE * (attempt + 1))
                        continue
                    raise

        call_metrics.success = False
        call_metrics.error_type = "all_models_exhausted"
        call_metrics.latency_ms = (time.perf_counter() - start) * 1000
        log_llm_call(call_metrics)
        raise RuntimeError("All Gemini models exhausted")


class AnthropicLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6",
                 timeout: float = DEFAULT_TIMEOUT_S, max_retries: int = DEFAULT_MAX_RETRIES):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self._model = model
        self._max_retries = max_retries

    def complete(self, system: str, messages: list[dict], max_tokens: int = 600) -> str:
        call_metrics = LLMCallMetrics(provider="Anthropic", model=self._model)
        start = time.perf_counter()

        for attempt in range(self._max_retries):
            try:
                resp = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                )
                result = "".join(block.text for block in resp.content if block.type == "text")
                call_metrics.tokens_prompt = getattr(resp.usage, "input_tokens", 0)
                call_metrics.tokens_completion = getattr(resp.usage, "output_tokens", 0)
                call_metrics.latency_ms = (time.perf_counter() - start) * 1000
                call_metrics.retries = attempt
                log_llm_call(call_metrics)
                return result
            except Exception as exc:
                if attempt < self._max_retries - 1:
                    call_metrics.retries = attempt + 1
                    time.sleep(DEFAULT_RETRY_BACKOFF_BASE * (attempt + 1))
                    continue
                call_metrics.success = False
                call_metrics.error_type = type(exc).__name__
                call_metrics.latency_ms = (time.perf_counter() - start) * 1000
                log_llm_call(call_metrics)
                raise

        call_metrics.success = False
        call_metrics.error_type = "max_retries_exceeded"
        call_metrics.latency_ms = (time.perf_counter() - start) * 1000
        log_llm_call(call_metrics)
        raise RuntimeError("Anthropic: max retries exceeded")


class MultiProviderFallbackLLMClient(LLMClient):
    """Chains multiple LLM providers. If provider 1 fails, automatically cascades to provider 2, 3, etc."""

    def __init__(self, clients: list[tuple[str, LLMClient]]):
        self.clients = clients
        self.mock = MockLLMClient()
        self._last_provider = ""
        self._last_model = ""

    @property
    def last_provider(self) -> str:
        return self._last_provider

    @property
    def last_model(self) -> str:
        return self._last_model

    def complete(self, system: str, messages: list[dict], max_tokens: int = 600) -> str:
        from app.observability import metrics as obs_metrics
        for idx, (provider_name, client) in enumerate(self.clients):
            try:
                res = client.complete(system, messages, max_tokens)
                if res and res.strip():
                    self._last_provider = provider_name
                    self._last_model = getattr(client, "_model", "unknown")
                    if idx > 0:
                        # This was a fallback
                        obs_metrics.increment("provider_fallback_used")
                    return res
            except Exception as exc:
                logger.warning("Provider %s failed (%s). Cascading to next available provider...", provider_name, exc)
                obs_metrics.increment(f"provider_{provider_name}_failures")
                continue

        logger.warning("All configured LLM providers failed. Falling back to MockLLMClient.")
        self._last_provider = "Mock"
        self._last_model = "mock"
        return self.mock.complete(system, messages, max_tokens)

    def complete_json(self, system: str, user_prompt: str, max_tokens: int = 500) -> dict:
        for provider_name, client in self.clients:
            try:
                res = client.complete_json(system, user_prompt, max_tokens)
                if res and isinstance(res, dict):
                    self._last_provider = provider_name
                    self._last_model = getattr(client, "_model", "unknown")
                    return res
            except Exception as exc:
                logger.warning("Provider %s complete_json failed (%s). Cascading...", provider_name, exc)
                continue

        return self.mock.complete_json(system, user_prompt, max_tokens)

    def complete_json_validated(
        self,
        system: str,
        user_prompt: str,
        schema_name: str,
        max_tokens: int = 500,
        max_retries: int = 2,
    ) -> dict:
        for provider_name, client in self.clients:
            try:
                res = client.complete_json_validated(system, user_prompt, schema_name, max_tokens, max_retries)
                if res and isinstance(res, dict):
                    self._last_provider = provider_name
                    self._last_model = getattr(client, "_model", "unknown")
                    return res
            except Exception as exc:
                logger.warning("Provider %s complete_json_validated failed (%s). Cascading...", provider_name, exc)
                continue

        return self.mock.complete_json_validated(system, user_prompt, schema_name, max_tokens, max_retries)


class MockLLMClient(LLMClient):
    """Deterministic stand-in so the full pipeline runs offline / without credentials."""

    _POSITIVE = {"great", "good", "happy", "excited", "love", "awesome", "nice", "thanks", "yay"}
    _NEGATIVE = {"sad", "bad", "angry", "upset", "frustrated", "tired", "stressed", "anxious", "hate", "worried"}

    def complete(self, system: str, messages: list[dict], max_tokens: int = 600) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        words = set(re.findall(r"[a-z']+", last_user.lower()))
        name_match = re.search(r"CHARACTER:\s*(\w+)", system)
        name = name_match.group(1) if name_match else "Nova"

        if words & self._NEGATIVE:
            opener = "Hey, that sounds rough."
        elif words & self._POSITIVE:
            opener = "Okay that's genuinely great, tell me more."
        else:
            opener = "Got it."

        if "REGENERATION FEEDBACK" in system:
            opener = f"Okay, let me try that differently — {opener.lower()}"

        mem_hint = ""
        if "MEMORIES" in system and "(none)" not in system.split("MEMORIES")[1][:50]:
            mem_hint = " (and yeah, I remembered what you told me before)"

        return f"[{name}] {opener}{mem_hint} — {_paraphrase(last_user)}"

    def complete_json(self, system: str, user_prompt: str, max_tokens: int = 500) -> dict:
        words = set(re.findall(r"[a-z']+", user_prompt.lower()))
        frustration = 0.7 if words & {"ugh", "annoyed", "frustrated", "angry"} else 0.1
        sadness = 0.6 if words & self._NEGATIVE else 0.05
        joy = 0.6 if words & self._POSITIVE else 0.1

        seed = int(hashlib.sha256(user_prompt.encode()).hexdigest(), 16) % 100 / 100

        if "emotion" in user_prompt.lower() or "intent" in user_prompt.lower():
            return {
                "emotion": {"joy": joy, "sadness": sadness, "frustration": frustration},
                "intensity": round(max(joy, sadness, frustration), 2),
                "intent": "venting" if sadness > 0.3 else ("sharing" if joy > 0.3 else "neutral"),
                "need": "validation" if sadness > 0.3 else ("celebration" if joy > 0.3 else "information"),
                "social_state": {
                    "openness": round(0.4 + seed * 0.3, 2),
                    "trust": round(0.4 + seed * 0.2, 2),
                    "irritation": round(frustration, 2),
                },
                "confidence": 0.5,
            }
        return {
            "emotional_fit": round(0.5 + seed * 0.3, 2),
            "persona_consistency": round(0.6 + seed * 0.2, 2),
            "context_relevance": round(0.6 + seed * 0.2, 2),
            "memory_consistency": 0.9,
            "repetition": round(0.3 + seed * 0.2, 2),
        }


def _paraphrase(text: str) -> str:
    text = text.strip()
    if len(text) > 60:
        text = text[:57] + "..."
    return f'you said "{text}"' if text else "what's on your mind?"


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def build_llm_client() -> LLMClient:
    """Builds a cascading multi-provider client with all configured API keys."""
    chain: list[tuple[str, LLMClient]] = []

    # 1. OpenRouter
    if settings.openrouter_api_key:
        chain.append(("OpenRouter", OpenRouterLLMClient(settings.openrouter_api_key, settings.openrouter_model)))

    # 2. Groq
    if settings.groq_api_key:
        chain.append(("Groq", GroqLLMClient(settings.groq_api_key, settings.groq_model)))

    # 3. Google Gemini
    if settings.gemini_api_key:
        chain.append(("Gemini", GeminiLLMClient(settings.gemini_api_key, settings.gemini_model)))

    # 4. Cohere
    if settings.cohere_api_key:
        chain.append(("Cohere", CohereLLMClient(settings.cohere_api_key, settings.cohere_model)))

    # 5. Anthropic
    if settings.anthropic_api_key:
        chain.append(("Anthropic", AnthropicLLMClient(settings.anthropic_api_key, settings.anthropic_model)))

    if chain:
        return MultiProviderFallbackLLMClient(chain)
    return MockLLMClient()
