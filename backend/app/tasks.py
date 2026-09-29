"""
The actual background task. Runs in a separate Celery worker process
(started with `celery -A app.celery_app worker`), not inside the FastAPI
request process — which means it can't reuse the request's SQLAlchemy
session (that session belongs to a different process's connection pool
entirely). Each task invocation opens and closes its own session.

This task wraps reflection.run_reflection() unchanged — nothing about
reflection.py itself is Celery-specific, which is exactly what its
docstring predicted back in V1 ("wrapping it as a Celery task is
mechanical").
"""
from app.celery_app import celery_app
from app.database import SessionLocal
from app.reflection import run_reflection


@celery_app.task(name="app.tasks.run_reflection_task")
def run_reflection_task(user_id: str) -> dict:
    db = SessionLocal()
    try:
        return run_reflection(db, user_id)
    finally:
        db.close()
