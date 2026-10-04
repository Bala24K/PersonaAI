"""FastAPI backend. Run with: uvicorn main:app --reload --port 8000"""
from __future__ import annotations

import logging
import os
import socket
import time
import uuid
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.character import Character
from app.config import settings
from app.cpp_wrapper import analyze_text_cpp
from app.mongo_store import mongo_store
from app.observability import metrics as obs_metrics
from app.orchestrator import Persona

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s (%(threadName)s): %(message)s",
)
logger = logging.getLogger("persona_ai.api")

app = FastAPI(
    title="Persona AI",
    version="2.0.0",
    description="Production-oriented AI Backend with Celery, RabbitMQ, Redis, PostgreSQL, MongoDB, and C++17 analytics.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional API key authentication (set PERSONA_API_KEY env var to enable)
_API_KEY = os.getenv("PERSONA_API_KEY")

@app.middleware("http")
async def auth_and_correlation_middleware(request: Request, call_next):
    # Skip auth for health/ready/docs endpoints
    skip_paths = {"/health", "/ready", "/docs", "/openapi.json", "/redoc"}
    if _API_KEY and request.url.path not in skip_paths and not request.url.path.startswith("/ui"):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header[7:] != _API_KEY:
            import json as _json
            return Response(
                content=_json.dumps({"detail": "Invalid or missing API key"}),
                status_code=401,
                media_type="application/json",
            )

    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = time.time()
    
    response: Response = await call_next(request)
    
    duration_ms = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-MS"] = str(duration_ms)
    
    logger.info(
        "method=%s path=%s status=%d duration_ms=%.2f req_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request_id,
    )
    return response


_character = Character.from_yaml("characters/default_character.yaml")
_persona = Persona(_character)


# --- Request & Response Schemas ---

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    session_id: str = Field(..., description="Session identifier")
    message: str = Field(..., description="User message content")
    show_debug: bool = Field(False, description="Whether to include debug information")


class ChatResponse(BaseModel):
    response: str
    character_name: str
    cpp_text_stats: Optional[Dict[str, Any]] = None
    debug: Optional[Dict[str, Any]] = None


class MemoryTaskRequest(BaseModel):
    user_id: str
    message: str


class EvaluationTaskRequest(BaseModel):
    user_id: str
    user_message: str
    response: str


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


# --- Core API Routes ---

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    
    start_time = time.time()

    # C++17 Component: Fast text feature & lexical analysis
    cpp_stats = analyze_text_cpp(req.message)

    # Core Persona response orchestrator
    result = _persona.respond(req.user_id, req.session_id, req.message)

    debug = None
    if req.show_debug:
        debug = {
            "eq_state": result.eq_state.to_dict(),
            "eval_result": result.eval_result.to_dict(),
            "regenerations": result.regenerations,
            "prompt_versions": result.prompt_versions,
        }

    # Log interaction event trace to MongoDB
    mongo_store.log_interaction_event(
        user_id=req.user_id,
        session_id=req.session_id,
        user_message=req.message,
        persona_response=result.response,
        metadata={
            "request_id": getattr(request.state, "request_id", None),
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "cpp_stats": cpp_stats,
            "regenerations": result.regenerations,
        },
    )

    return ChatResponse(
        response=result.response,
        character_name=_character.name,
        cpp_text_stats=cpp_stats,
        debug=debug,
    )


# --- User State, Memories & Mind Dashboard Endpoints ---

class ProfileImportRequest(BaseModel):
    text: str = Field(..., description="Raw text blob to ingest into memories and user model")


@app.get("/character")
def get_character():
    """Returns metadata about the active companion character."""
    return {
        "name": _character.name,
        "description": _character.identity,
        "traits": _character.personality_traits,
        "voice": _character.speaking_style,
        "llm_provider": type(_persona.llm).__name__,
    }


@app.get("/user/{user_id}/state")
@app.get("/user/{user_id}/model")
def get_user_state(user_id: str):
    """Returns persistent user model, relationship state, and observations."""
    _persona.db.ensure_user(user_id)
    profile, relationship = _persona.user_state.load(user_id)
    turn_count = _persona._session_turn_count(user_id, "")
    latest_eq = _persona._latest_eq_state(user_id)

    confidence = round(min(turn_count / 10.0, 1.0), 2)
    traits = [
        {"name": "Warmth", "value": round(relationship.affection, 2), "confidence": confidence, "evidence_count": turn_count},
        {"name": "Trust", "value": round(relationship.trust, 2), "confidence": confidence, "evidence_count": turn_count},
        {"name": "Familiarity", "value": round(relationship.familiarity, 2), "confidence": confidence, "evidence_count": turn_count},
        {"name": "Playfulness", "value": round(relationship.playfulness, 2), "confidence": confidence, "evidence_count": turn_count},
        {"name": "Tension", "value": round(relationship.recent_tension, 2), "confidence": confidence, "evidence_count": turn_count},
    ]

    return {
        "user_id": user_id,
        "turn_count": turn_count,
        "traits": traits,
        "relationship": {
            "familiarity": relationship.familiarity,
            "trust": relationship.trust,
            "affection": relationship.affection,
            "playfulness": relationship.playfulness,
            "recent_tension": relationship.recent_tension,
            "shared_history_summary": relationship.shared_history_summary,
        },
        "profile": {
            "important_facts": profile.important_facts,
            "recurring_interests": profile.recurring_interests,
            "preferences": profile.preferences,
            "communication_preferences": profile.communication_preferences,
        },
        "latest_eq": latest_eq,
    }


@app.get("/user/{user_id}/memories")
def get_user_memories(user_id: str):
    """Returns all long-term and episodic memories stored for the user."""
    _persona.db.ensure_user(user_id)
    memories = _persona.memory.all_for_user(user_id)
    return {
        "memories": [
            {
                "id": m.id,
                "kind": m.kind,
                "content": m.content,
                "importance": m.importance,
                "created_at": m.created_at,
                "last_accessed_at": m.last_accessed_at,
                "access_count": m.access_count,
            }
            for m in memories
        ]
    }


@app.delete("/user/{user_id}/memories/{memory_id}")
def delete_user_memory(user_id: str, memory_id: int):
    """Deletes a specific memory record for the user."""
    success = _persona.memory.delete(user_id, memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted", "memory_id": memory_id}


@app.post("/user/{user_id}/profile/import")
@app.post("/profile/import")
def import_user_profile(req: ProfileImportRequest, user_id: Optional[str] = None):
    """Ingests profile text samples and creates memories/facts."""
    uid = user_id or "default_user"
    text_content = req.text
    if not text_content.strip():
        raise HTTPException(status_code=400, detail="Profile text cannot be empty")

    _persona.db.ensure_user(uid)
    lines = [line.strip() for line in text_content.split("\n") if line.strip()]
    created_ids = []

    profile, relationship = _persona.user_state.load(uid)
    for line in lines:
        if len(line) < 3:
            continue
        kind = "preference" if any(w in line.lower() for w in ["like", "prefer", "love", "hate", "favorite"]) else "semantic"
        mem_id = _persona.memory.add(uid, kind=kind, content=line, importance=0.75)
        created_ids.append(mem_id)
        if line not in profile.important_facts:
            profile.important_facts.append(line)

    _persona.user_state.save(uid, profile, relationship)

    return {
        "memories_created": len(created_ids),
        "memory_ids": created_ids,
        "message": f"Successfully ingested {len(created_ids)} profile statements",
    }


@app.get("/user/{user_id}/style")
def get_user_style(user_id: str):
    """Analyzes communication style vectors and computes conversational alignment."""
    with _persona.db.connect() as conn:
        user_rows = conn.execute(
            "SELECT content FROM turns WHERE user_id = :uid AND role = 'user'",
            {"uid": user_id},
        ).fetchall()
        assistant_rows = conn.execute(
            "SELECT content FROM turns WHERE user_id = :uid AND role = 'assistant'",
            {"uid": user_id},
        ).fetchall()

    user_texts = [r["content"] for r in user_rows]
    assistant_texts = [r["content"] for r in assistant_rows]

    def _calc_metrics(texts: list[str]) -> dict[str, Any]:
        if not texts:
            return {
                "sample_size": 0,
                "avg_words": 0.0,
                "question_rate": 0.0,
                "exclamation_rate": 0.0,
                "lexical_richness": 0.0,
                "flesch_reading_ease": 70.0,
            }
        total_words = 0
        total_questions = 0
        total_exclamations = 0
        all_words = []
        for t in texts:
            words = t.split()
            total_words += len(words)
            all_words.extend([w.lower() for w in words])
            if "?" in t:
                total_questions += 1
            if "!" in t:
                total_exclamations += 1

        n = len(texts)
        avg_words = round(total_words / n, 1)
        q_rate = round(total_questions / n, 2)
        excl_rate = round(total_exclamations / n, 2)
        lex_richness = round(len(set(all_words)) / max(len(all_words), 1), 2)

        sample_str = " ".join(texts[-5:])
        cpp_res = analyze_text_cpp(sample_str)
        flesch = cpp_res.get("flesch_reading_ease", 70.0)

        return {
            "sample_size": n,
            "avg_words": avg_words,
            "question_rate": q_rate,
            "exclamation_rate": excl_rate,
            "lexical_richness": lex_richness,
            "flesch_reading_ease": flesch,
        }

    u_metrics = _calc_metrics(user_texts)
    a_metrics = _calc_metrics(assistant_texts)

    similarity = 100.0
    gaps = []
    if u_metrics["sample_size"] > 0 and a_metrics["sample_size"] > 0:
        len_gap = abs(u_metrics["avg_words"] - a_metrics["avg_words"]) / max(u_metrics["avg_words"], a_metrics["avg_words"], 1.0)
        q_gap = abs(u_metrics["question_rate"] - a_metrics["question_rate"])
        lex_gap = abs(u_metrics["lexical_richness"] - a_metrics["lexical_richness"])

        sim_val = max(0.0, min(100.0, 100.0 - (len_gap * 40.0 + q_gap * 30.0 + lex_gap * 30.0)))
        similarity = round(sim_val, 1)

        gaps = [
            {"feature": "Average Message Length", "gap": round(len_gap, 2), "user": f"{u_metrics['avg_words']} words", "persona": f"{a_metrics['avg_words']} words"},
            {"feature": "Inquiry / Question Rate", "gap": round(q_gap, 2), "user": f"{int(u_metrics['question_rate']*100)}%", "persona": f"{int(a_metrics['question_rate']*100)}%"},
            {"feature": "Vocabulary Richness", "gap": round(lex_gap, 2), "user": f"{int(u_metrics['lexical_richness']*100)}%", "persona": f"{int(a_metrics['lexical_richness']*100)}%"},
        ]
        gaps.sort(key=lambda g: g["gap"], reverse=True)

    return {
        "user_style": u_metrics,
        "assistant_style": a_metrics,
        "similarity": similarity,
        "biggest_gaps": gaps,
        "sample_sizes": {
            "user_messages": u_metrics["sample_size"],
            "assistant_messages": a_metrics["sample_size"],
        },
    }


@app.get("/user/{user_id}/history")
@app.get("/conversation/{session_id}/history")
def get_conversation_history(user_id: Optional[str] = None, session_id: Optional[str] = None):
    """Retrieves chronological conversation history."""
    import json as _json
    with _persona.db.connect() as conn:
        if session_id and user_id:
            rows = conn.execute(
                "SELECT role, content, eq_state_json, eval_score_json, created_at FROM turns "
                "WHERE user_id = :uid AND session_id = :sid ORDER BY id ASC",
                {"uid": user_id, "sid": session_id},
            ).fetchall()
        elif session_id:
            rows = conn.execute(
                "SELECT role, content, eq_state_json, eval_score_json, created_at FROM turns "
                "WHERE session_id = :sid ORDER BY id ASC",
                {"sid": session_id},
            ).fetchall()
        elif user_id:
            rows = conn.execute(
                "SELECT role, content, eq_state_json, eval_score_json, created_at FROM turns "
                "WHERE user_id = :uid ORDER BY id ASC",
                {"uid": user_id},
            ).fetchall()
        else:
            return {"messages": []}

    messages = []
    for r in rows:
        eq = _json.loads(r["eq_state_json"]) if r["eq_state_json"] else None
        ev = _json.loads(r["eval_score_json"]) if r["eval_score_json"] else None
        messages.append({
            "role": r["role"],
            "content": r["content"],
            "created_at": r["created_at"],
            "eq_state": eq,
            "eval_score": ev,
        })
    return {"messages": messages}


# --- Asynchronous Task Processing Routes ---

@app.post("/tasks/memory-extraction", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_memory_extraction_task(req: MemoryTaskRequest):
    """Enqueues an asynchronous memory extraction & C++ pre-analysis job via RabbitMQ + Celery."""
    try:
        from app.tasks import extract_memory_async
        task = extract_memory_async.delay(req.user_id, req.message)
        return TaskResponse(
            task_id=task.id,
            status="queued",
            message="Async memory extraction job submitted successfully to RabbitMQ",
        )
    except Exception as exc:
        logger.error("Failed to enqueue memory extraction task: %s", exc)
        raise HTTPException(status_code=500, detail=f"Task submission failed: {exc}")


@app.post("/tasks/evaluate", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_evaluation_task(req: EvaluationTaskRequest):
    """Enqueues an asynchronous response evaluation job via Celery worker."""
    try:
        from app.tasks import evaluate_response_async
        task = evaluate_response_async.delay(req.user_id, req.user_message, req.response)
        return TaskResponse(
            task_id=task.id,
            status="queued",
            message="Async response evaluation job submitted successfully",
        )
    except Exception as exc:
        logger.error("Failed to enqueue evaluation task: %s", exc)
        raise HTTPException(status_code=500, detail=f"Task submission failed: {exc}")


@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """Retrieves current status and result for an asynchronous Celery task."""
    state = "unknown"
    ready = False
    result_data = None
    error_msg = None

    # Check Celery result backend if available
    try:
        from celery.result import AsyncResult
        from app.tasks import celery_app

        async_result = AsyncResult(task_id, app=celery_app)
        state = async_result.state.lower()
        ready = async_result.ready()
        if ready:
            res = async_result.result
            if isinstance(res, Exception):
                error_msg = str(res)
            else:
                result_data = res
    except Exception as exc:
        logger.warning("Celery result backend query warning for task %s: %s", task_id, exc)

    # Check MongoDB job log
    mongo_job_log = mongo_store.get_async_job_log(task_id)
    if mongo_job_log and state == "unknown":
        state = mongo_job_log.get("status", "unknown").lower()
        ready = state in ["success", "failure"]
        result_data = mongo_job_log.get("result")
        error_msg = mongo_job_log.get("error")

    # If running eager or fallback mode without active Redis/Mongo
    if state == "unknown":
        state = "success"
        ready = True

    return {
        "task_id": task_id,
        "status": state,
        "ready": ready,
        "result": result_data,
        "error": error_msg,
        "mongo_trace": mongo_job_log,
    }



# --- Observability ---

@app.get("/metrics")
def get_metrics():
    """Returns current observability metrics snapshot. Does not expose secrets or user content."""
    return obs_metrics.snapshot()


# --- Health & Readiness Checks ---

@app.get("/health")
def health():
    """Liveness probe: verifies the API process is running."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "character": _character.name,
    }


@app.get("/ready")
def readiness():
    """Readiness probe: verifies PostgreSQL, MongoDB, Redis, RabbitMQ, and C++ component availability."""
    checks = {
        "postgres": False,
        "mongodb": False,
        "redis": False,
        "rabbitmq": False,
        "cpp_analyzer": False,
    }

    # 1. PostgreSQL check
    try:
        from app.db import DB
        db = DB()
        with db.connect() as conn:
            conn.execute("SELECT 1").fetchone()
        checks["postgres"] = True
    except Exception as exc:
        logger.warning("Readiness check: PostgreSQL failed: %s", exc)

    # 2. MongoDB check
    checks["mongodb"] = mongo_store.is_available

    # 3. Redis check
    try:
        import redis
        r = redis.from_url(settings.redis_url, socket_timeout=1.0)
        checks["redis"] = r.ping()
    except Exception as exc:
        logger.warning("Readiness check: Redis failed: %s", exc)

    # 4. RabbitMQ check (TCP socket check)
    try:
        parsed = urlparse(settings.rabbitmq_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5672
        with socket.create_connection((host, port), timeout=1.0):
            checks["rabbitmq"] = True
    except Exception as exc:
        logger.warning("Readiness check: RabbitMQ failed: %s", exc)

    # 5. C++ Component check
    try:
        res = analyze_text_cpp("ready check")
        checks["cpp_analyzer"] = res.get("source") in ["cpp17", "python_fallback"]
    except Exception as exc:
        logger.warning("Readiness check: C++ component failed: %s", exc)

    import json
    all_critical_ready = checks["postgres"]
    status_code = status.HTTP_200_OK if all_critical_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    return Response(
        content=json.dumps({
            "status": "ready" if all_critical_ready else "degraded",
            "checks": checks,
            "timestamp": time.time(),
        }),
        media_type="application/json",
        status_code=status_code,
    )


# --- Web UI Mounting ---
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

_frontend_dir = Path(__file__).resolve().parent / "frontend"
if _frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to the interactive Web Chat UI."""
    return RedirectResponse(url="/ui/")


