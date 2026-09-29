"""
Pluggable generation backend.

If ANTHROPIC_API_KEY is set in the environment, `generate()` calls the
real Claude API with the assembled system prompt + conversation history.

If it is not set, `generate()` falls back to `_template_response`: a
small, honest, rule-based responder built from the *same* strategy/emotion/
memory signals a real LLM would receive. This means the entire pipeline —
memory extraction, retrieval, user modeling, emotion/strategy selection,
personalization — is runnable and demoable with zero API keys and zero
external network access. Response *quality* without a real LLM is
obviously limited; that trade-off is the whole point of keeping this
fallback path honest rather than faking something more impressive.

Swap point: nothing else in the codebase calls the Anthropic API
directly or knows whether the fallback is active — everything goes
through `generate()`.
"""
from __future__ import annotations

import requests

from app.config import (
    ANTHROPIC_API_KEY, ANTHROPIC_API_URL, ANTHROPIC_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
)

STRATEGY_OPENERS = {
    "listen_first": [
        "That sounds like a lot. I'm here — say more if you want.",
        "Yeah, that's rough. I'm listening.",
    ],
    "acknowledge_then_offer_choice": [
        "That's frustrating, for real. Do you want to vent for a bit, or actually dig into what happened?",
        "Ugh, sorry — that one stings. Want to talk it through or just get it off your chest?",
    ],
    "help_prioritize": [
        "Okay, that's a lot at once. Let's break it down — what's actually due first?",
        "Sounds overloaded. What's the one thing that, if handled, would take the most pressure off?",
    ],
    "ask_clarifying_question": [
        "Before I say anything useful — what's actually making you unsure here?",
        "What outcome are you hoping for, ideally?",
    ],
    "celebrate": [
        "Let's go — that's genuinely great, congrats.",
        "Hell yes, that's a big deal. How are you feeling about it?",
    ],
    "answer_directly": [],
}


def has_llm_key() -> bool:
    return bool(ANTHROPIC_API_KEY or GEMINI_API_KEY)


def _call_gemini_generate(system_prompt: str, messages: list[dict], model: str = GEMINI_MODEL) -> str | None:
    if not GEMINI_API_KEY:
        return None
    import time
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"

    gemini_contents = []
    for msg in messages:
        role = "model" if msg.get("role") == "assistant" else "user"
        content = msg.get("content", "")
        if isinstance(content, str):
            gemini_contents.append({"role": role, "parts": [{"text": content}]})
        elif isinstance(content, list):
            text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and "text" in b]
            if text_parts:
                gemini_contents.append({"role": role, "parts": [{"text": "\n".join(text_parts)}]})

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": gemini_contents
    }

    # Retry up to 3 attempts for transient 429 rate limits or network blips
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 429 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                text_blocks = [p.get("text", "") for p in parts if "text" in p]
                return "\n".join(text_blocks).strip()
            return None
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            print(f"Gemini API call failed: {e}")
            return None
    return None


def _template_response(user_text: str, memories: list[str], strategy: str) -> str:
    """
    Deterministic offline fallback. Picks an opener based on the chosen
    strategy, optionally references a retrieved memory to demonstrate
    personalization, and otherwise stays honest about being a
    placeholder rather than pretending to be a full conversational model.
    """
    import random

    openers = STRATEGY_OPENERS.get(strategy, [])
    opener = random.choice(openers) if openers else None

    parts = []
    if opener:
        parts.append(opener)

    if memories:
        parts.append(f"(Also — this connects to something you mentioned before: \"{memories[0]}\".)")

    if not opener:
        if has_llm_key():
            parts.append(
                "[The connected LLM API did not return a response for this message. "
                "The request was likely blocked by the LLM's automated safety filters or experienced a temporary API timeout.]"
            )
        else:
            parts.append(
                "I don't have a real language model connected right now (no ANTHROPIC_API_KEY or GEMINI_API_KEY set), "
                "so this is a placeholder response generated from the strategy/memory pipeline instead of "
                "an actual model. Set the API key to get real conversational responses."
            )

    return " ".join(parts)


def generate(system_prompt: str, conversation_history: list[dict], user_text: str,
             memories: list[str], strategy: str) -> str:
    """
    conversation_history: list of {"role": "user"|"assistant", "content": str}
    Returns the assistant's reply text.
    """
    if not has_llm_key():
        return _template_response(user_text, memories, strategy)

    messages = conversation_history + [{"role": "user", "content": user_text}]

    if GEMINI_API_KEY:
        gemini_reply = _call_gemini_generate(system_prompt, messages, GEMINI_MODEL)
        if gemini_reply:
            return gemini_reply

    if ANTHROPIC_API_KEY:
        try:
            resp = requests.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 600,
                    "system": system_prompt,
                    "messages": messages,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
            return "\n".join(text_blocks).strip() or _template_response(user_text, memories, strategy)
        except requests.RequestException as e:
            return (
                f"[LLM call failed: {e}. Falling back to template response.] "
                + _template_response(user_text, memories, strategy)
            )

    return _template_response(user_text, memories, strategy)


def complete(system_prompt: str, user_text: str, max_tokens: int = 400) -> str | None:
    """
    A bare system+user -> raw text call, with no conversation history and
    no template fallback — used for structured-output tasks like memory
    extraction (see memory/llm_extraction.py), not for conversational
    replies. Returns None if no API key is configured or the call fails
    for any reason, so callers can fall back to a non-LLM path rather
    than crash.
    """
    if not has_llm_key():
        return None

    messages = [{"role": "user", "content": user_text}]
    if GEMINI_API_KEY:
        gemini_reply = _call_gemini_generate(system_prompt, messages, GEMINI_MODEL)
        if gemini_reply:
            return gemini_reply

    if ANTHROPIC_API_KEY:
        try:
            resp = requests.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": max_tokens,
                    "system": system_prompt,
                    "messages": messages,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
            text = "\n".join(text_blocks).strip()
            return text or None
        except requests.RequestException:
            return None

    return None


def _raw_call(system_prompt: str, messages: list[dict], tools: list[dict], max_tokens: int) -> dict | None:
    try:
        resp = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": messages,
                "tools": tools,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def generate_with_tools(system_prompt: str, conversation_history: list[dict], user_text: str,
                         db, user_id: str, max_rounds: int = 3) -> dict | None:
    """
    Agentic variant of generate(): gives the model the search_memory tool
    (app/llm/tools.py) and lets it decide whether and how many times to
    call it, rather than always retrieving top-K memories before every
    call (see agents/orchestrator.py's "always" vs "agentic" modes).

    Returns None if no API key is configured — callers (orchestrator)
    fall back to the classic always-retrieve flow in that case, the same
    "None means fall back" convention memory/llm_extraction.py uses.
    Otherwise returns {"reply": str, "retrieved_memories": list[str],
    "retrieved_memory_ids": list[str], "tool_calls": int}.
    """
    if not ANTHROPIC_API_KEY:
        return None

    from app.llm.tools import TOOL_DEFINITIONS, execute_tool

    messages = conversation_history + [{"role": "user", "content": user_text}]
    all_retrieved: list[str] = []
    all_retrieved_ids: list[str] = []
    tool_call_count = 0

    for _ in range(max_rounds):
        data = _raw_call(system_prompt, messages, TOOL_DEFINITIONS, max_tokens=600)
        if data is None:
            return None

        content_blocks = data.get("content", [])
        tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]

        if not tool_use_blocks:
            text = "\n".join(b["text"] for b in content_blocks if b.get("type") == "text").strip()
            return {
                "reply": text, "retrieved_memories": all_retrieved,
                "retrieved_memory_ids": all_retrieved_ids, "tool_calls": tool_call_count,
            }

        # Model wants to call one or more tools: execute each, append the
        # assistant's tool-use turn and our tool-result turn, then loop.
        messages.append({"role": "assistant", "content": content_blocks})
        tool_results = []
        for block in tool_use_blocks:
            tool_call_count += 1
            result_text, memory_texts, memory_ids = execute_tool(block["name"], block.get("input", {}), db, user_id)
            all_retrieved.extend(memory_texts)
            all_retrieved_ids.extend(memory_ids)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": result_text,
            })
        messages.append({"role": "user", "content": tool_results})

    # Exhausted max_rounds without a final text response — return
    # whatever memories were gathered along the way rather than nothing.
    return {
        "reply": "", "retrieved_memories": all_retrieved,
        "retrieved_memory_ids": all_retrieved_ids, "tool_calls": tool_call_count,
    }
