"""
Celery application instance.

This is what design doc Phase 11/12's "the slow brain belongs on a
background job queue" actually looks like as running infrastructure,
not just a comment saying it should exist. See tasks.py for the actual
task and docs/BACKGROUND_JOBS.md for the exact commands used to install
Redis, start a worker, and verify a task really executed on it.

Kept deliberately minimal: one task (reflection), no task routing,
no retries/rate limits configured — those are the natural next additions
once there's more than one kind of background job to run.
"""
from celery import Celery

from app.config import RABBITMQ_URL, REDIS_URL

celery_app = Celery("persona", broker=RABBITMQ_URL, backend=REDIS_URL, include=["app.tasks"])
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    result_expires=3600,  # poll results are kept for an hour
)

