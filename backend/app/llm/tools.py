"""
Tool definitions for agentic memory retrieval (design doc Phase 9).

V1-V3 always retrieve top-K memories before every LLM call, whether or
not the message actually needs them (see agents/orchestrator.py step 3).
This module is the alternative the design doc originally asked for:
expose memory search as a tool and let the model decide whether and how
many times to call it — a simple "what's my name" doesn't need a memory
search, but "remind me what I said about the interview last week" might
need two or three searches with different phrasings.

This is opt-in (config.MEMORY_MODE="agentic", default remains "always")
specifically because it changes a real tradeoff, not just an
implementation detail: agentic mode costs an extra model round-trip
whenever a tool is actually called, and shifts retrieval quality from
"guaranteed, every turn" to "whatever the model decides it needs" — a
worse choice for a malfunctioning or very small model, a better one for
a capable model on messages where blind top-K retrieval would just add
irrelevant noise to the prompt.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.memory.store import retrieve_relevant

TOOL_DEFINITIONS = [
    {
        "name": "search_memory",
        "description": (
            "Search this person's long-term memory for facts, past events, "
            "preferences, or relationships relevant to the current message. "
            "Call this when their message references something you might "
            "already know about them, or when recalling prior context would "
            "change how you respond. Don't call it for messages that are "
            "self-contained (greetings, questions with no personal context, "
            "one-off factual questions)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for — a short phrase describing the information needed, not necessarily the person's exact wording.",
                }
            },
            "required": ["query"],
        },
    }
]


def execute_tool(name: str, tool_input: dict, db: Session, user_id: str) -> tuple[str, list[str], list[str]]:
    """
    Runs a tool call and returns (text_for_model, raw_memory_texts,
    memory_ids). raw_memory_texts/memory_ids are surfaced separately so
    the API/UI can show what was actually retrieved (same shape as the
    "always" mode's retrieved_memories field) and so the assistant
    message can record which memory IDs were used — see
    database.py::Message.retrieved_memory_ids_json, which is what lets
    a "wrong_memory" feedback tag trace back to a specific memory
    instead of a vague complaint.
    """
    if name == "search_memory":
        query = tool_input.get("query", "")
        rows = retrieve_relevant(db, user_id, query)
        if not rows:
            return "No relevant memories found.", [], []
        texts = [r.content for r in rows]
        ids = [r.id for r in rows]
        formatted = "\n".join(f"- {t}" for t in texts)
        return formatted, texts, ids

    return f"Unknown tool: {name}", [], []
