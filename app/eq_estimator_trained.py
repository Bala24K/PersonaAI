"""Trained-model backend for the EQ estimator (see app/ml/eq_model.py for why this is
sklearn rather than a fine-tuned transformer in this sandbox).

The underlying PersonaEQModel scores a single message in isolation, so temporal
continuity (S_t depends on S_{t-1}, per Section 4) is implemented explicitly here via
exponential smoothing on the numeric outputs, rather than being learned end-to-end the
way a real recurrent/attention encoder over (S_{t-1}, C_t) would do it. That's a real
architectural simplification versus the pitch doc, not just a smaller model — noting it
plainly so it isn't mistaken for the same thing.
"""
from __future__ import annotations

import os

from app.eq_estimator import EQState
from app.ml.eq_model import PersonaEQModel

# how much weight the previous state keeps each turn (0 = no memory, 1 = never update)
SMOOTHING_ALPHA = 0.35


class TrainedEQEstimator:
    def __init__(self, model_path: str = "models/eq_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained EQ model at {model_path}. Run `python scripts/train_eq_model.py` first."
            )
        self._model = PersonaEQModel.load(model_path)

    def update(self, recent_turns: list[dict], current_message: str, previous_state: EQState) -> EQState:
        raw = self._model.predict(current_message)

        intensity = SMOOTHING_ALPHA * previous_state.intensity + (1 - SMOOTHING_ALPHA) * raw["intensity"]

        social = {}
        for key in ("openness", "trust", "irritation"):
            prev_v = previous_state.social_state.get(key, 0.5)
            new_v = raw["social_state"][key]
            social[key] = round(SMOOTHING_ALPHA * prev_v + (1 - SMOOTHING_ALPHA) * new_v, 3)

        return EQState(
            emotion={str(k): float(v) for k, v in raw["emotion"].items()},
            intensity=round(intensity, 3),
            intent=str(raw["intent"]),
            need=str(raw["need"]),
            social_state=social,
            confidence=float(raw["confidence"]),
        )


def build_eq_estimator(llm=None):
    """Factory respecting EQ_BACKEND env var: 'trained' (default if model exists) or 'llm'."""
    backend = os.getenv("EQ_BACKEND", "auto")
    model_path = os.getenv("EQ_MODEL_PATH", "models/eq_model.pkl")

    if backend == "llm":
        from app.eq_estimator import EQEstimator
        assert llm is not None, "llm client required for EQ_BACKEND=llm"
        return EQEstimator(llm)

    if backend == "trained" or (backend == "auto" and os.path.exists(model_path)):
        return TrainedEQEstimator(model_path)

    from app.eq_estimator import EQEstimator
    assert llm is not None, "llm client required when no trained model is available"
    return EQEstimator(llm)
