"""
Pure data types for the user model, with zero database dependency.

Split out from model.py specifically so code that only needs the
*shape* of a trait estimate — not database access — doesn't have to
import SQLAlchemy at all. Concretely: ml/evaluation/eval_critic.py
constructs TraitEstimate objects directly (to test critic.py's
cliche-aversion check) in an environment that may only have
ml/requirements.txt installed, with no SQLAlchemy present. Before this
split, importing TraitEstimate meant importing all of model.py, which
meant importing app.database, which meant importing sqlalchemy —
caught by the evaluation harness's own isolated-venv fresh-install
test, the same test that's caught a real dependency gap every round
it's been run.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TraitEstimate:
    name: str
    value: float
    confidence: float
    evidence_count: int
