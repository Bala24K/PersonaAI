"""Character Model — persistent, user-independent persona definition.

Maps to Section 8 of the pitch doc. Loaded from a YAML file so non-engineers can author
and tweak characters without touching code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class Character:
    name: str
    identity: str
    personality_traits: list[str] = field(default_factory=list)
    backstory: str = ""
    values: list[str] = field(default_factory=list)
    speaking_style: str = ""
    emotional_tendencies: str = ""
    behavioral_rules: list[str] = field(default_factory=list)
    relationship_behavior: str = ""
    roleplay_behavior: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> "Character":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            name=data["name"],
            identity=data.get("identity", ""),
            personality_traits=data.get("personality_traits", []),
            backstory=data.get("backstory", ""),
            values=data.get("values", []),
            speaking_style=data.get("speaking_style", ""),
            emotional_tendencies=data.get("emotional_tendencies", ""),
            behavioral_rules=data.get("behavioral_rules", []),
            relationship_behavior=data.get("relationship_behavior", ""),
            roleplay_behavior=data.get("roleplay_behavior", ""),
        )

    def to_system_block(self) -> str:
        """Render the character definition as the CHARACTER section of the system prompt."""
        rules = "\n".join(f"  - {r}" for r in self.behavioral_rules) or "  - (none specified)"
        traits = ", ".join(self.personality_traits) or "(none specified)"
        values = ", ".join(self.values) or "(none specified)"
        return (
            f"# CHARACTER: {self.name}\n"
            f"Identity: {self.identity}\n"
            f"Personality traits: {traits}\n"
            f"Backstory: {self.backstory}\n"
            f"Values: {values}\n"
            f"Speaking style: {self.speaking_style}\n"
            f"Emotional tendencies: {self.emotional_tendencies}\n"
            f"Behavioral rules:\n{rules}\n"
            f"Relationship behavior: {self.relationship_behavior}\n"
            f"Role-play behavior: {self.roleplay_behavior}\n"
        )
