"""Main LLM Generator (Section 9 of pitch doc), pluggable and multi-provider resilient so
the system gracefully falls back across providers (OpenRouter, Groq, Cohere, Gemini, Anthropic)
down to the deterministic mock responder if external services encounter outages or rate limits.
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

from app.config import settings

logger = logging.getLogger("persona_ai.llm")


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


class OpenRouterLLMClient(LLMClient):
    """Calls OpenRouter chat completions API (OpenAI-compatible)."""

    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.3-70b-instruct"):
        self._api_key = api_key
        self._model = model

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

        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()


class GroqLLMClient(LLMClient):
    """Calls Groq Cloud API (OpenAI-compatible, ultra low-latency)."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self._api_key = api_key
        self._model = model

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
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()


class CohereLLMClient(LLMClient):
    """Calls Cohere v2 Chat API."""

    def __init__(self, api_key: str, model: str = "command-r-plus"):
        self._api_key = api_key
        self._model = model

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
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["message"]["content"][0]["text"].strip()


class GeminiLLMClient(LLMClient):
    """Calls Google Gemini REST API using standard urllib with exponential retries."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self._api_key = api_key
        self._model = model

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

        for current_model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={self._api_key}"
            req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with urllib.request.urlopen(req, timeout=25) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        candidates = data.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"].get("parts", [])
                            text_blocks = [p.get("text", "") for p in parts if "text" in p]
                            return "\n".join(text_blocks).strip()
                        return ""
                except urllib.error.HTTPError as err:
                    if err.code == 404:
                        break
                    if err.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    raise
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    raise

        raise RuntimeError("All Gemini models exhausted")


class AnthropicLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system: str, messages: list[dict], max_tokens: int = 600) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class MultiProviderFallbackLLMClient(LLMClient):
    """Chains multiple LLM providers. If provider 1 fails, automatically cascades to provider 2, 3, etc."""

    def __init__(self, clients: list[tuple[str, LLMClient]]):
        self.clients = clients
        self.mock = MockLLMClient()

    def complete(self, system: str, messages: list[dict], max_tokens: int = 600) -> str:
        for provider_name, client in self.clients:
            try:
                res = client.complete(system, messages, max_tokens)
                if res and res.strip():
                    return res
            except Exception as exc:
                logger.warning("Provider %s failed (%s). Cascading to next available provider...", provider_name, exc)
                continue

        logger.warning("All configured LLM providers failed. Falling back to MockLLMClient.")
        return self.mock.complete(system, messages, max_tokens)

    def complete_json(self, system: str, user_prompt: str, max_tokens: int = 500) -> dict:
        for provider_name, client in self.clients:
            try:
                res = client.complete_json(system, user_prompt, max_tokens)
                if res and isinstance(res, dict):
                    return res
            except Exception as exc:
                logger.warning("Provider %s complete_json failed (%s). Cascading...", provider_name, exc)
                continue

        return self.mock.complete_json(system, user_prompt, max_tokens)


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
