"""Role-play Scene State (Section 12 of pitch doc). Only injected into the prompt when
the user is actively in a role-play scene, to avoid biasing normal conversation."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from app.db import DB


@dataclass
class SceneState:
    active: bool = False
    location: str = ""
    time_of_scene: str = ""
    characters_present: list[str] = field(default_factory=list)
    current_action: str = ""
    recent_events: list[str] = field(default_factory=list)
    unresolved_threads: list[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        if not self.active:
            return ""
        events = "\n".join(f"  - {e}" for e in self.recent_events[-6:]) or "  - (scene just started)"
        threads = ", ".join(self.unresolved_threads) or "(none)"
        chars = ", ".join(self.characters_present) or "(just you two)"
        return (
            f"# SCENE STATE (active role-play — stay consistent with this)\n"
            f"Location: {self.location or 'unspecified'}\n"
            f"Time: {self.time_of_scene or 'unspecified'}\n"
            f"Characters present: {chars}\n"
            f"Current action: {self.current_action or 'unspecified'}\n"
            f"Recent events:\n{events}\n"
            f"Unresolved threads: {threads}\n"
        )


class SceneStateStore:
    def __init__(self, db: DB):
        self._db = db

    def load(self, user_id: str) -> SceneState:
        with self._db.connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM scene_state WHERE user_id = :uid", {"uid": user_id}
            ).fetchone()
        if not row:
            return SceneState()
        return SceneState(**json.loads(row["state_json"]))

    def save(self, user_id: str, scene: SceneState):
        with self._db.connect() as conn:
            conn.execute(
                "INSERT INTO scene_state (user_id, state_json, updated_at) VALUES (:uid, :state, :ts) "
                "ON CONFLICT(user_id) DO UPDATE SET state_json = excluded.state_json, "
                "updated_at = excluded.updated_at",
                {"uid": user_id, "state": json.dumps(scene.__dict__), "ts": time.time()},
            )
