"""
The "fast brain" (design doc Phase 11): the synchronous per-message
pipeline that has to run before we can reply at all.

    analyze -> retrieve -> plan -> generate -> personalize -> store

The "slow brain" (deeper reflection, contradiction detection, richer
memory extraction) lives in app/reflection.py and is triggered after the
turn completes, not blocking the reply — see main.py's /chat route.
"""
from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.config import ANTHROPIC_API_KEY, CRITIC_MODE, MEMORY_MODE
from app.critic import describe_issues, heuristic_critique, llm_critique
from app.database import Conversation, Message
from app.emotion.engine import analyze as analyze_emotion
from app.llm.client import generate, generate_with_tools, has_llm_key
from app.llm.personalize import apply as personalize
from app.llm.prompts import build_system_prompt
from app.memory.store import ingest_text, retrieve_relevant
from app.user_model.model import snapshot, update_trait
from app.user_model.signals import extract_signals

HISTORY_WINDOW = 12  # messages of raw recent context sent to the LLM


def _recent_history(db: Session, conversation_id: str) -> list[dict]:
    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
        .all()
    )
    tail = msgs[-HISTORY_WINDOW:]
    return [{"role": m.role, "content": m.content} for m in tail]


def handle_user_message(db: Session, user_id: str, conversation_id: str, text: str) -> dict:
    """
    Runs the full fast-path pipeline for one incoming user message and
    returns everything the API needs to respond and everything the
    reflection job needs to do deeper analysis later.
    """
    # 1. Analyze --------------------------------------------------------
    emotion_result = analyze_emotion(text)
    signals = extract_signals(text)

    # 2. Fast user-model update (lightweight, per-message) ---------------
    # A fuller pass happens in reflection.py; this keeps traits reasonably
    # live even before that runs.
    for trait_name, (observed_value, weight) in signals.items():
        update_trait(db, user_id, trait_name, observed_value, weight)
    trait_estimates = {e.name: e for e in snapshot(db, user_id)}

    # 3-5. Retrieve + Plan + Generate ---------------------------------------
    # "agentic" mode (config.MEMORY_MODE) skips the upfront retrieval and
    # instead gives the model a search_memory tool, letting it decide
    # whether/how many times to call it — see llm/tools.py and
    # llm/client.py::generate_with_tools. Falls back to the classic
    # always-retrieve path automatically if no API key is configured
    # (generate_with_tools returns None in that case) or if MEMORY_MODE
    # is left at its default "always".
    strategy = emotion_result.strategy
    history = _recent_history(db, conversation_id)
    agentic_result = None

    if MEMORY_MODE == "agentic":
        system_prompt_agentic = build_system_prompt(
            traits=trait_estimates, memories=[], strategy=strategy,
            situation=emotion_result.situation, dominant_emotion=emotion_result.dominant,
        )
        agentic_result = generate_with_tools(system_prompt_agentic, history, text, db, user_id)

    if agentic_result is not None:
        raw_reply = agentic_result["reply"]
        memory_texts = agentic_result["retrieved_memories"]
        memory_ids = agentic_result["retrieved_memory_ids"]
        tool_calls = agentic_result["tool_calls"]
    else:
        memory_rows = retrieve_relevant(db, user_id, text)
        memory_texts = [m.content for m in memory_rows]
        memory_ids = [m.id for m in memory_rows]
        system_prompt = build_system_prompt(
            traits=trait_estimates, memories=memory_texts, strategy=strategy,
            situation=emotion_result.situation, dominant_emotion=emotion_result.dominant,
        )
        raw_reply = generate(system_prompt, history, text, memory_texts, strategy)
        tool_calls = 0

    # 6. Personalization layer --------------------------------------------
    final_reply = personalize(raw_reply, trait_estimates)

    # 6.5. Critic (design doc Phase 12) -------------------------------------
    # Checks the reply against the strategy it was supposed to follow.
    # CRITIC_MODE="heuristic" (default) always runs this — it's pure text
    # matching, zero API cost. If issues are found AND a real model is
    # configured, one regeneration attempt is made with the critique
    # folded into the system prompt; never more than one, to bound cost
    # and latency. The offline template responder has no way to act on
    # feedback, so regeneration is skipped when there's no API key —
    # the critic still runs and reports issues either way, which is
    # useful for seeing it work even without a live model.
    critique_issues: list[str] = []
    regenerated = False
    if CRITIC_MODE != "off":
        critique_issues = heuristic_critique(final_reply, strategy, emotion_result.situation, trait_estimates)

        if not critique_issues and CRITIC_MODE == "heuristic_llm":
            llm_result = llm_critique(final_reply, strategy, emotion_result.situation, emotion_result.dominant)
            if llm_result is not None and not llm_result["ok"]:
                critique_issues = [llm_result["reason"] or "llm_critique_flagged"]

        if critique_issues and has_llm_key():
            feedback_note = describe_issues(critique_issues)
            revised_system_prompt = (
                build_system_prompt(
                    traits=trait_estimates, memories=memory_texts, strategy=strategy,
                    situation=emotion_result.situation, dominant_emotion=emotion_result.dominant,
                )
                + f"\n\nA previous draft of this reply had a problem: {feedback_note} Do not repeat that mistake."
            )
            revised_raw = generate(revised_system_prompt, history, text, memory_texts, strategy)
            final_reply = personalize(revised_raw, trait_estimates)
            regenerated = True

    # 7. Store --------------------------------------------------------------
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=text,
        emotion_json=json.dumps(emotion_result.scores),
        situation=emotion_result.situation,
        strategy=strategy,
    )
    db.add(user_msg)
    assistant_msg = Message(
        conversation_id=conversation_id, role="assistant", content=final_reply,
        retrieved_memory_ids_json=json.dumps(memory_ids),
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(user_msg)
    db.refresh(assistant_msg)

    # Cheap memory ingestion runs inline (heuristic, not an LLM call);
    # deeper extraction happens in reflection.py.
    ingest_text(db, user_id, text, source="conversation")

    return {
        "reply": final_reply,
        "emotion": emotion_result.scores,
        "dominant_emotion": emotion_result.dominant,
        "situation": emotion_result.situation,
        "strategy": strategy,
        "flat_affect_flag": emotion_result.flat_affect_flag,
        "trained_model_used": emotion_result.trained_model_used,
        "retrieved_memories": memory_texts,
        "memory_mode": "agentic" if agentic_result is not None else "always",
        "tool_calls": tool_calls,
        "critique_issues": critique_issues,
        "regenerated": regenerated,
        "user_message_id": user_msg.id,
        "assistant_message_id": assistant_msg.id,
    }


def get_or_create_conversation(db: Session, user_id: str, conversation_id: str | None) -> Conversation:
    if conversation_id:
        convo = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if convo:
            return convo
    convo = Conversation(user_id=user_id)
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return convo
