"""Persistence layer, backed by SQLAlchemy Core so the same code runs against SQLite
(zero-config local dev, the default) or Postgres (set DATABASE_URL, per the stack
recommendation in Section 14 of the pitch doc) without any call-site changes.

Deliberately Core, not ORM: the query patterns here are simple enough that ORM overhead
buys nothing, and Core keeps the SQL visible and easy to reason about.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager

from sqlalchemy import (
    Column, Float, Integer, MetaData, String, Table, Text, create_engine, text,
)

DEFAULT_SQLITE_PATH = "data/persona.db"

metadata = MetaData()

users = Table(
    "users", metadata,
    Column("user_id", String, primary_key=True),
    Column("created_at", Float, nullable=False),
)

user_state_table = Table(
    "user_state", metadata,
    Column("user_id", String, primary_key=True),
    Column("profile_json", Text, nullable=False),
    Column("emotional_tendencies_json", Text, nullable=False),
    Column("relationship_json", Text, nullable=False),
    Column("updated_at", Float, nullable=False),
)

memories_table = Table(
    "memories", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String, nullable=False),
    Column("kind", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("importance", Float, nullable=False, default=0.5),
    Column("created_at", Float, nullable=False),
    Column("last_accessed_at", Float, nullable=False),
    Column("access_count", Integer, nullable=False, default=0),
)

turns_table = Table(
    "turns", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String, nullable=False),
    Column("session_id", String, nullable=False),
    Column("role", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("eq_state_json", Text, nullable=True),
    Column("eval_score_json", Text, nullable=True),
    Column("created_at", Float, nullable=False),
)

scene_state_table = Table(
    "scene_state", metadata,
    Column("user_id", String, primary_key=True),
    Column("state_json", Text, nullable=False),
    Column("updated_at", Float, nullable=False),
)

# Every generation attempt (accepted or rejected-by-evaluator), so later we can build
# real DPO preference pairs (chosen vs rejected for the same context) per Section 10.
candidates_table = Table(
    "candidates", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String, nullable=False),
    Column("session_id", String, nullable=False),
    Column("user_message", Text, nullable=False),
    Column("system_prompt", Text, nullable=False),
    Column("response", Text, nullable=False),
    Column("eval_json", Text, nullable=False),
    Column("accepted", Integer, nullable=False),  # 0/1, portable across sqlite/postgres
    Column("created_at", Float, nullable=False),
)


def default_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    db_path = os.getenv("PERSONA_DB_PATH", DEFAULT_SQLITE_PATH)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    return f"sqlite:///{db_path}"


class ConnWrapper:
    """Thin wrapper so call sites can do conn.execute(sql_str, {...}).fetchone()/.fetchall()
    with dict-like row access (row["col"]), regardless of SQLite vs Postgres underneath.
    """

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql: str, params: dict | None = None):
        return self._conn.execute(text(sql), params or {}).mappings()


class DB:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or default_database_url()
        connect_args = {"check_same_thread": False} if self.database_url.startswith("sqlite") else {}
        try:
            self.engine = create_engine(self.database_url, connect_args=connect_args, future=True)
            metadata.create_all(self.engine)
        except Exception:
            if not self.database_url.startswith("sqlite"):
                fallback_path = os.getenv("PERSONA_DB_PATH", DEFAULT_SQLITE_PATH)
                os.makedirs(os.path.dirname(fallback_path) or ".", exist_ok=True)
                self.database_url = f"sqlite:///{fallback_path}"
                self.engine = create_engine(self.database_url, connect_args={"check_same_thread": False}, future=True)
                metadata.create_all(self.engine)
            else:
                raise


    @contextmanager
    def connect(self):
        conn = self.engine.connect()
        trans = conn.begin()
        try:
            yield ConnWrapper(conn)
            trans.commit()
        except Exception:
            trans.rollback()
            raise
        finally:
            conn.close()

    def ensure_user(self, user_id: str):
        with self.connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM users WHERE user_id = :uid", {"uid": user_id}
            ).fetchone()
            if not exists:
                conn.execute(
                    "INSERT INTO users (user_id, created_at) VALUES (:uid, :ts)",
                    {"uid": user_id, "ts": time.time()},
                )
            state_exists = conn.execute(
                "SELECT 1 FROM user_state WHERE user_id = :uid", {"uid": user_id}
            ).fetchone()
            if not state_exists:
                conn.execute(
                    "INSERT INTO user_state (user_id, profile_json, emotional_tendencies_json, "
                    "relationship_json, updated_at) VALUES (:uid, :profile, :tendencies, :rel, :ts)",
                    {
                        "uid": user_id,
                        "profile": "{}",
                        "tendencies": "{}",
                        "rel": (
                            '{"familiarity": 0.05, "trust": 0.3, "affection": 0.1, '
                            '"playfulness": 0.3, "recent_tension": 0.0, "shared_history_summary": ""}'
                        ),
                        "ts": time.time(),
                    },
                )
