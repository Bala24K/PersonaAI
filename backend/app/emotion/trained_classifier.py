"""
Loads the trained emotion classifier (see ml/training/train_emotion_classifier.py
for how it was produced) and exposes one function: predict(text) -> dict of
{label: probability} over the six classes the model was actually trained on:

    neutral, anxiety, joy, sadness, surprise, anger

This is a real trained model (TF-IDF + Logistic Regression, ~40k labeled
tweets, evaluated with a held-out test set — see trained/training_report.txt
for the actual numbers: 40.2% accuracy / 0.333 macro F1 on a 6-class
problem, beating the 16.7% random and ~33% majority-class baselines but
nowhere near production-grade). It does not know about fear, disgust, or
frustration — the training data had no good source for those — which is
exactly why emotion/engine.py keeps the lexicon active as a complementary
signal rather than replacing it outright. See docs/ARCHITECTURE.md for
the full reasoning.

Loading is lazy and cached: the joblib file is only read once, on first
call to predict(), not at import time, so importing this module has no
cost if the trained-model path ends up unused.
"""
from __future__ import annotations

from pathlib import Path

import joblib

_MODEL_PATH = Path(__file__).resolve().parent / "trained" / "emotion_classifier.joblib"

# The exact label set the model was trained on (see
# ml/training/train_emotion_classifier.py). Kept as an explicit constant
# here, not derived from the loaded pipeline, so callers (engine.py) can
# reason about which categories the trained model *could* have an opinion
# on even before the model is loaded.
TRAINED_LABELS = ("neutral", "anxiety", "joy", "sadness", "surprise", "anger")

_pipeline = None
_load_attempted = False
_load_error: str | None = None


def _get_pipeline():
    global _pipeline, _load_attempted, _load_error
    if _load_attempted:
        return _pipeline
    _load_attempted = True
    try:
        _pipeline = joblib.load(_MODEL_PATH)
    except Exception as e:  # noqa: BLE001 - want to degrade gracefully either way
        _load_error = str(e)
        _pipeline = None
    return _pipeline


def is_available() -> bool:
    return _get_pipeline() is not None


def load_error() -> str | None:
    """For diagnostics: why the trained model isn't available, if it isn't."""
    _get_pipeline()
    return _load_error


def predict(text: str) -> dict[str, float]:
    """Returns {} if the model failed to load; otherwise a probability
    distribution over the six trained classes."""
    pipeline = _get_pipeline()
    if pipeline is None or not text or not text.strip():
        return {}
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    return {cls: float(p) for cls, p in zip(classes, proba)}
