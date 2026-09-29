"""
MongoDB Document/Event Storage Layer.

Responsibility:
Stores unstructured interaction traces, async job metadata, telemetry logs,
and evaluation benchmark records.

PostgreSQL handles structured relational state (users, sessions, memories, traits),
while MongoDB stores append-only, flexible document events and execution traces.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Configurable Mongo URL & DB
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "persona_ai")

class MongoStore:
    def __init__(self, mongo_url: Optional[str] = None, db_name: Optional[str] = None):
        self.mongo_url = mongo_url or MONGODB_URL
        self.db_name = db_name or MONGODB_DB_NAME
        self._client = None
        self._db = None
        self._available = False
        self._init_connection()

    def _init_connection(self) -> None:
        try:
            import pymongo
            self._client = pymongo.MongoClient(self.mongo_url, serverSelectionTimeoutMS=2000)
            # Ping database to verify active connection
            self._client.admin.command('ping')
            self._db = self._client[self.db_name]
            self._available = True
            self._ensure_indexes()
            logger.info("Successfully connected to MongoDB at %s (DB: %s)", self.mongo_url, self.db_name)
        except Exception as exc:
            self._available = False
            self._client = None
            self._db = None
            logger.warning("MongoDB connection unavailable (%s). Operating in graceful fallback mode.", exc)

    def _ensure_indexes(self) -> None:
        if not self._available or self._db is None:
            return
        try:
            self._db.interaction_events.create_index([("user_id", 1), ("created_at", -1)])
            self._db.async_job_logs.create_index([("task_id", 1)], unique=True)
            self._db.async_job_logs.create_index([("user_id", 1), ("created_at", -1)])
            self._db.evaluation_records.create_index([("eval_type", 1), ("created_at", -1)])
        except Exception as exc:
            logger.warning("Failed to create Mongo indexes: %s", exc)

    @property
    def is_available(self) -> bool:
        return self._available

    def log_interaction_event(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        persona_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Logs chat interaction telemetry document."""
        if not self._available or self._db is None:
            return None
        try:
            doc = {
                "user_id": user_id,
                "session_id": session_id,
                "user_message": user_message,
                "persona_response": persona_response,
                "created_at": time.time(),
                "metadata": metadata or {},
            }
            res = self._db.interaction_events.insert_one(doc)
            return str(res.inserted_id)
        except Exception as exc:
            logger.error("Error inserting interaction event to Mongo: %s", exc)
            return None

    def log_async_job(
        self,
        task_id: str,
        task_name: str,
        user_id: str,
        status: str,
        duration_s: float = 0.0,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[str]:
        """Logs or updates an async Celery task execution trace document."""
        if not self._available or self._db is None:
            return None
        try:
            filter_doc = {"task_id": task_id}
            update_doc = {
                "$set": {
                    "task_name": task_name,
                    "user_id": user_id,
                    "status": status,
                    "duration_s": duration_s,
                    "result": result or {},
                    "error": error,
                    "updated_at": time.time(),
                },
                "$setOnInsert": {
                    "task_id": task_id,
                    "created_at": time.time(),
                }
            }
            res = self._db.async_job_logs.update_one(filter_doc, update_doc, upsert=True)
            return task_id
        except Exception as exc:
            logger.error("Error logging async job to Mongo: %s", exc)
            return None

    def log_evaluation_record(
        self,
        eval_type: str,
        metrics: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Logs evaluation/benchmark run record document."""
        if not self._available or self._db is None:
            return None
        try:
            doc = {
                "eval_type": eval_type,
                "metrics": metrics,
                "details": details or {},
                "created_at": time.time(),
            }
            res = self._db.evaluation_records.insert_one(doc)
            return str(res.inserted_id)
        except Exception as exc:
            logger.error("Error logging evaluation record to Mongo: %s", exc)
            return None

    def get_interaction_events(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves interaction events for a given user."""
        if not self._available or self._db is None:
            return []
        try:
            cursor = self._db.interaction_events.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
            events = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                events.append(doc)
            return events
        except Exception as exc:
            logger.error("Error querying interaction events from Mongo: %s", exc)
            return []

    def get_async_job_log(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves async job metadata document by task ID."""
        if not self._available or self._db is None:
            return None
        try:
            doc = self._db.async_job_logs.find_one({"task_id": task_id})
            if doc:
                doc["_id"] = str(doc["_id"])
            return doc
        except Exception as exc:
            logger.error("Error querying async job log from Mongo: %s", exc)
            return None

# Singleton instance
mongo_store = MongoStore()
