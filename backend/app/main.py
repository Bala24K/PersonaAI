"""
API layer. Routes map directly onto the design doc's Phase 1 sketch:

    POST /chat
    POST /profile/import
    GET  /user/{id}/model
    GET  /conversation/{id}/history
    POST /feedback

Run with:  uvicorn app.main:app --reload --port 8000   (from backend/)
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.agents.orchestrator import get_or_create_conversation, handle_user_message
from app.config import BASE_DIR, REFLECTION_MODE
from app.database import (
    Conversation, Feedback, Memory, Message, User, get_session, init_db, new_id,
)
from app.memory.store import ingest_text, load_index_from_db
from app.reflection import run_reflection
from app.schemas import (
    ChatRequest, ChatResponse, CreateUserResponse, FeedbackRequest, ProfileImportRequest,
)
from app.user_model.feedback_learning import apply_feedback, flag_wrong_memory
from app.user_model.model import snapshot

app = FastAPI(title="Persona AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()
    db = next(get_session())
    try:
        load_index_from_db(db)
    finally:
        db.close()


@app.get("/")
def root():
    return RedirectResponse(url="/ui/")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/user", response_model=CreateUserResponse)
def create_user(db: Session = Depends(get_session)):
    user = User(id=new_id())
    db.add(user)
    db.commit()
    return CreateUserResponse(user_id=user.id)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_session)):
    user = db.query(User).filter(User.id == req.user_id).first()
    if not user:
        user = User(id=req.user_id)
        db.add(user)
        db.commit()

    convo = get_or_create_conversation(db, req.user_id, req.conversation_id)
    result = handle_user_message(db, req.user_id, convo.id, req.message)

    # Slow-brain pass. REFLECTION_MODE="inline" (default) runs it
    # synchronously here, same as V1-V3 — simplest, and keeps trend_note
    # available in this same response. "background" dispatches it as a
    # real Celery task instead and returns a task_id to poll via
    # GET /reflection/{task_id} — see app/tasks.py and
    # docs/BACKGROUND_JOBS.md for how this was verified against a live
    # worker + Redis instance.
    if REFLECTION_MODE == "background":
        from app.tasks import run_reflection_task
        task = run_reflection_task.delay(req.user_id)
        reflection = {"status": "queued", "task_id": task.id}
    else:
        reflection = run_reflection(db, req.user_id)

    return ChatResponse(
        conversation_id=convo.id,
        reply=result["reply"],
        dominant_emotion=result["dominant_emotion"],
        emotion_scores=result["emotion"],
        situation=result["situation"],
        strategy=result["strategy"],
        flat_affect_flag=result["flat_affect_flag"],
        trained_model_used=result["trained_model_used"],
        retrieved_memories=result["retrieved_memories"],
        memory_mode=result["memory_mode"],
        tool_calls=result["tool_calls"],
        critique_issues=result["critique_issues"],
        regenerated=result["regenerated"],
        reflection=reflection,
        user_message_id=result["user_message_id"],
        assistant_message_id=result["assistant_message_id"],
    )


@app.get("/reflection/{task_id}")
def get_reflection_result(task_id: str):
    """
    Poll for a background reflection task's result — only meaningful
    when REFLECTION_MODE="background" (see /chat). Backed by Celery's
    result backend (Redis), not application state, so this works
    correctly even if a different FastAPI worker process than the one
    that dispatched the task handles the poll.
    """
    from celery.result import AsyncResult
    from app.celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)
    if result.state == "PENDING":
        return {"status": "pending", "result": None}
    if result.state == "SUCCESS":
        return {"status": "done", "result": result.result}
    if result.state == "FAILURE":
        return {"status": "failed", "result": str(result.info)}
    return {"status": result.state.lower(), "result": None}


@app.post("/profile/import")
def import_profile(req: ProfileImportRequest, db: Session = Depends(get_session)):
    """
    Phase 4: user-authorized profile ingestion. Accepts a raw text blob
    (bio, writing sample, exported notes — whatever the person chooses to
    paste in) and runs it through the same extraction pipeline used for
    chat messages, tagged with source="profile_import" so it's traceable
    back to where a memory came from.
    """
    user = db.query(User).filter(User.id == req.user_id).first()
    if not user:
        user = User(id=req.user_id)
        db.add(user)
        db.commit()

    # Split on paragraphs/sentences so extraction runs per-chunk rather
    # than treating a whole document as one blob.
    chunks = [c.strip() for c in req.text.split("\n") if c.strip()]
    created = []
    for chunk in chunks:
        created += ingest_text(db, req.user_id, chunk, source="profile_import")

    return {
        "memories_created": len(created),
        "memories": [{"type": m.type, "content": m.content, "importance": m.importance} for m in created],
    }


@app.get("/user/{user_id}/model")
def get_user_model(user_id: str, db: Session = Depends(get_session)):
    traits = snapshot(db, user_id)
    return {
        "traits": [
            {"name": t.name, "value": t.value, "confidence": t.confidence, "evidence_count": t.evidence_count}
            for t in traits
        ]
    }


@app.get("/user/{user_id}/style")
def get_user_style(user_id: str, db: Session = Depends(get_session)):
    """
    Design doc Phase 6 ("personality imitation" / writing-style
    analysis): computes a StyleVector for the person's own messages and
    for the AI's replies to them, and reports how closely the two
    match. Both vectors are computed across every conversation this
    user has had, not just the current one — a single conversation is
    often too short (see `sample_size` in the response) for the rarer
    features (emoji rate, question frequency) to be meaningful.
    """
    from app.user_model.style_vector import biggest_style_gaps, compute_style_vector, style_similarity

    user_messages = (
        db.query(Message)
        .join(Message.conversation)
        .filter(Message.role == "user")
        .filter(Message.conversation.has(user_id=user_id))
        .all()
    )
    assistant_messages = (
        db.query(Message)
        .join(Message.conversation)
        .filter(Message.role == "assistant")
        .filter(Message.conversation.has(user_id=user_id))
        .all()
    )

    user_vector = compute_style_vector([m.content for m in user_messages])
    assistant_vector = compute_style_vector([m.content for m in assistant_messages])

    similarity = None
    gaps = []
    if user_vector.sample_size > 0 and assistant_vector.sample_size > 0:
        similarity = style_similarity(user_vector, assistant_vector)
        gaps = [{"feature": name, "gap": round(gap, 3)} for name, gap in biggest_style_gaps(user_vector, assistant_vector)]

    return {
        "user_style": vars(user_vector),
        "assistant_style": vars(assistant_vector),
        "similarity": round(similarity, 3) if similarity is not None else None,
        "biggest_gaps": gaps,
        "note": (
            "Similarity is None until both sides have at least one message. "
            "Treat small sample_size (under ~10 messages per side) as low-confidence — "
            "these are frequency-based features that need enough text to stabilize."
        ),
    }


@app.get("/user/{user_id}/drift")
def get_user_drift(user_id: str, db: Session = Depends(get_session)):
    """
    Design doc V4 ("personality drift"): reports traits whose estimate
    has meaningfully and durably shifted over the course of this user's
    history, rather than treating whatever was learned first as
    permanent truth. See user_model/drift.py for the detection method
    and its three thresholds.
    """
    from app.user_model.drift import detect_drift

    findings = detect_drift(db, user_id)
    return {
        "drift_detected": len(findings) > 0,
        "findings": [
            {
                "trait": f.trait,
                "early_value": f.early_value,
                "recent_value": f.recent_value,
                "change": f.change,
                "direction": f.direction,
                "recent_stability": f.recent_stability,
                "observations": {"early": f.early_n, "recent": f.recent_n},
                "summary": f.describe(),
            }
            for f in findings
        ],
        "note": (
            "An empty findings list is a real answer, not a failure — most users' "
            "traits don't durably shift. Drift requires a change of at least 0.15 "
            "between window means, at least 5 observations per window, AND a settled "
            "recent window (stddev under 0.15), so erratic users don't generate "
            "constant false reports."
        ),
    }


@app.get("/user/{user_id}/trait-timeline/{trait_name}")
def get_trait_timeline(user_id: str, trait_name: str, db: Session = Depends(get_session)):
    """Raw value-over-time series for one trait, for charting drift."""
    from app.user_model.drift import trait_timeline

    return {"trait": trait_name, "points": trait_timeline(db, user_id, trait_name)}


@app.post("/user/{user_id}/drift/apply")
def apply_user_drift(user_id: str, db: Session = Depends(get_session)):
    """Active drift rebalancing: recenters current trait estimates to recent window means."""
    from app.user_model.drift import recenter_drifted_traits

    rebalanced = recenter_drifted_traits(db, user_id)
    return {
        "applied": len(rebalanced) > 0,
        "rebalanced_traits": rebalanced,
        "note": "Re-centered durable trait shifts to align baseline user expectations."
    }


@app.get("/user/{user_id}/memories")
def get_user_memories(user_id: str, db: Session = Depends(get_session)):
    rows = (
        db.query(Memory)
        .filter(Memory.user_id == user_id)
        .order_by(Memory.importance.desc())
        .all()
    )
    return {
        "memories": [
            {
                "id": m.id, "type": m.type, "topic": m.topic, "content": m.content,
                "importance": m.importance, "source": m.source,
                "access_count": m.access_count,
            }
            for m in rows
        ]
    }


@app.delete("/user/{user_id}/memories/{memory_id}")
def delete_memory(user_id: str, memory_id: str, db: Session = Depends(get_session)):
    """The person stays in control of what the system believes about them
    (design doc section 17) — memories can be deleted outright."""
    row = db.query(Memory).filter(Memory.id == memory_id, Memory.user_id == user_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Memory not found")
    db.delete(row)
    db.commit()
    return {"deleted": memory_id}


@app.get("/conversation/{conversation_id}/history")
def get_history(conversation_id: str, db: Session = Depends(get_session)):
    convo = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "messages": [
            {"role": m.role, "content": m.content, "emotion": m.emotion(),
             "situation": m.situation, "strategy": m.strategy}
            for m in convo.messages
        ]
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest, db: Session = Depends(get_session)):
    msg = db.query(Message).filter(Message.id == req.message_id).first()
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")
    row = Feedback(
        message_id=req.message_id, user_id=req.user_id,
        rating=req.rating, tags_json=json.dumps(req.tags),
    )
    db.add(row)
    db.commit()

    # Phase 10, closed: a down-vote's tags nudge the specific trait they're
    # actually about, immediately — see user_model/feedback_learning.py.
    result = apply_feedback(db, req.user_id, req.rating, req.tags)

    # "wrong_memory" traces back to the actual memories retrieved for
    # this specific message (via Message.retrieved_memory_ids_json) and
    # flags them for review, instead of falling through to
    # unhandled_tags with no further action.
    if req.rating == "down" and "wrong_memory" in req.tags:
        flagged_ids = flag_wrong_memory(db, msg)
        result["flagged_memories"] = flagged_ids
        if "wrong_memory" in result.get("unhandled_tags", []):
            result["unhandled_tags"].remove("wrong_memory")

    return {"stored": True, **result}


# Serve the chat UI at /ui (kept separate from the API namespace at "/"
# so both can coexist without route collisions). Visit http://localhost:8000/ui/
_frontend_dir = BASE_DIR / "frontend"
if _frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
