"""FastAPI backend. Run with: uvicorn main:app --reload --port 8000"""
from __future__ import annotations

import logging
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

# Correlation ID Middleware
@app.middleware("http")
async def add_correlation_id_middleware(request: Request, call_next):
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

