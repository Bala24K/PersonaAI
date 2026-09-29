"""
Asynchronous Task Processing Engine (Celery + RabbitMQ + Redis + MongoDB).

Architecture:
- RabbitMQ: Message/task broker ensuring reliable queueing and acknowledgement.
- Redis: Result backend and short-lived execution state storage.
- Celery: Asynchronous worker execution with retries, exponential backoff, and late ack semantics.
- MongoDB: Task execution logging and evaluation telemetry storage.
"""
from __future__ import annotations

import logging
import time
from celery import Celery
from celery.schedules import crontab

from app.config import settings
from app.mongo_store import mongo_store

logger = logging.getLogger(__name__)

celery_app = Celery(
    "persona_ai",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    result_expires=3600,
    beat_schedule={
        "consolidate-memory-nightly": {
            "task": "app.tasks.consolidate_all_users",
            "schedule": crontab(hour=3, minute=0),
        },
        "export-training-data-weekly": {
            "task": "app.tasks.export_training_data",
            "schedule": crontab(day_of_week=0, hour=4, minute=0),
        },
    },
)


@celery_app.task(
    name="app.tasks.consolidate_memory",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def consolidate_memory(self, user_id: str) -> dict:
    """Summarizes a user's episodic memories into RelationshipState.shared_history_summary."""
    start_time = time.time()
    try:
        from app.db import DB
        from app.memory import MemoryStore
        from app.user_state import UserStateStore

        db = DB()
        memory_store = MemoryStore(db)
        user_state_store = UserStateStore(db)

        memories = memory_store.all_for_user(user_id)
        if not memories:
            res = {"user_id": user_id, "status": "no_memories"}
            mongo_store.log_async_job(
                task_id=self.request.id or "sync",
                task_name="consolidate_memory",
                user_id=user_id,
                status="SUCCESS",
                duration_s=time.time() - start_time,
                result=res,
            )
            return res

        top = sorted(memories, key=lambda m: (m.importance, m.last_accessed_at), reverse=True)[:8]
        summary = " | ".join(m.content for m in top)

        profile, relationship = user_state_store.load(user_id)
        relationship.shared_history_summary = summary[:1000]
        user_state_store.save(user_id, profile, relationship)

        res = {"user_id": user_id, "status": "consolidated", "memories_used": len(top)}
        mongo_store.log_async_job(
            task_id=self.request.id or "sync",
            task_name="consolidate_memory",
            user_id=user_id,
            status="SUCCESS",
            duration_s=time.time() - start_time,
            result=res,
        )
        return res
    except Exception as exc:
        mongo_store.log_async_job(
            task_id=self.request.id or "sync",
            task_name="consolidate_memory",
            user_id=user_id,
            status="FAILURE",
            duration_s=time.time() - start_time,
            error=str(exc),
        )
        raise


@celery_app.task(
    name="app.tasks.extract_memory_async",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def extract_memory_async(self, user_id: str, message: str) -> dict:
    """Asynchronously performs C++ text feature extraction and memory ingestion."""
    start_time = time.time()
    try:
        from app.cpp_wrapper import analyze_text_cpp
        from app.db import DB
        from app.memory import MemoryStore

        # Run C++17 feature analysis
        cpp_stats = analyze_text_cpp(message)

        db = DB()
        memory_store = MemoryStore(db)

        # Simple semantic extraction check
        memories_added = 0
        if len(message) > 15:
            kind = "episodic" if any(w in message.lower() for w in ["feel", "stressed", "happy", "angry", "sad"]) else "semantic"
            importance = 0.8 if kind == "episodic" else 0.5
            memory_store.add(user_id, kind, f"Extracted statement: {message[:100]}", importance=importance)
            memories_added += 1

        result = {
            "user_id": user_id,
            "memories_added": memories_added,
            "cpp_text_stats": cpp_stats,
            "processed_at": time.time(),
        }

        mongo_store.log_async_job(
            task_id=self.request.id or "sync",
            task_name="extract_memory_async",
            user_id=user_id,
            status="SUCCESS",
            duration_s=time.time() - start_time,
            result=result,
        )
        return result
    except Exception as exc:
        mongo_store.log_async_job(
            task_id=self.request.id or "sync",
            task_name="extract_memory_async",
            user_id=user_id,
            status="FAILURE",
            duration_s=time.time() - start_time,
            error=str(exc),
        )
        raise


@celery_app.task(
    name="app.tasks.evaluate_response_async",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def evaluate_response_async(self, user_id: str, user_message: str, response: str) -> dict:
    """Asynchronously evaluates a response candidate and logs evaluation telemetry."""
    start_time = time.time()
    try:
        from app.character import Character
        from app.eq_estimator import EQState
        from app.evaluator import ResponseEvaluator
        from app.llm_client import build_llm_client

        character = Character.from_yaml("characters/default_character.yaml")
        llm = build_llm_client()
        evaluator = ResponseEvaluator(llm)

        eval_res = evaluator.evaluate(
            character=character,
            eq_state=EQState.initial(),
            recent_assistant_turns=[],
            candidate=response,
        )

        res_dict = {
            "overall": eval_res.overall,
            "emotional_fit": eval_res.emotional_fit,
            "persona_consistency": eval_res.persona_consistency,
            "context_relevance": eval_res.context_relevance,
            "memory_consistency": eval_res.memory_consistency,
            "repetition": eval_res.repetition,
            "is_acceptable": eval_res.overall >= settings.min_acceptable_score,
        }



        mongo_store.log_async_job(
            task_id=self.request.id or "sync",
            task_name="evaluate_response_async",
            user_id=user_id,
            status="SUCCESS",
            duration_s=time.time() - start_time,
            result=res_dict,
        )
        return res_dict
    except Exception as exc:
        mongo_store.log_async_job(
            task_id=self.request.id or "sync",
            task_name="evaluate_response_async",
            user_id=user_id,
            status="FAILURE",
            duration_s=time.time() - start_time,
            error=str(exc),
        )
        raise


@celery_app.task(name="app.tasks.consolidate_all_users")
def consolidate_all_users() -> dict:
    from app.db import DB

    db = DB()
    with db.connect() as conn:
        rows = conn.execute("SELECT user_id FROM users").fetchall()
    user_ids = [r["user_id"] for r in rows]
    for uid in user_ids:
        consolidate_memory.delay(uid)
    return {"users_queued": len(user_ids)}


@celery_app.task(name="app.tasks.export_training_data")
def export_training_data() -> dict:
    """Triggers the SFT/DPO dataset export as a scheduled batch job."""
    from training.export_dataset import export_dpo, export_sft
    from app.db import DB

    db = DB()
    sft_count = export_sft(db, "data/sft_dataset.jsonl")
    dpo_count = export_dpo(db, "data/dpo_dataset.jsonl")
    result = {"sft_examples": sft_count, "dpo_pairs": dpo_count}

    mongo_store.log_evaluation_record(
        eval_type="dataset_export",
        metrics=result,
    )
    return result
