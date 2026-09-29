"""Real, trained multi-task EQ model.

This is the honest stand-in for Section 3's "Transformer encoder (DeBERTa/RoBERTa/
ModernBERT)": this sandbox has no access to a model hub and no working GPU-capable torch
install (both confirmed by hitting actual errors, not assumed), so a true transformer
fine-tune isn't buildable here. This IS a real trained multi-task model — shared TF-IDF
features feeding per-task heads, trained end-to-end on bootstrap_dataset.py — with the
exact same input/output contract as EQEstimator, so it's a genuine drop-in and the
swap to a real encoder later touches only this file.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from app.ml.bootstrap_dataset import Example, generate


@dataclass
class EvalMetrics:
    emotion_accuracy: float
    intent_accuracy: float
    need_accuracy: float
    intensity_mae: float
    social_state_mae: float


class PersonaEQModel:
    """Shared TF-IDF representation ("shared Transformer" stand-in) + per-task heads,
    mirroring the multi-task architecture in Section 5 even though the shared encoder
    itself is TF-IDF rather than a learned contextual embedding.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=4000)
        self.emotion_head = LogisticRegression(max_iter=1000)
        self.intent_head = LogisticRegression(max_iter=1000)
        self.need_head = LogisticRegression(max_iter=1000)
        self.intensity_head = Ridge(alpha=1.0)
        # social_state = [openness, trust, irritation], jointly regressed
        self.social_head = MultiOutputRegressor(Ridge(alpha=1.0))

    def fit(self, examples: list[Example]):
        texts = [e.text for e in examples]
        X = self.vectorizer.fit_transform(texts)

        self.emotion_head.fit(X, [e.emotion for e in examples])
        self.intent_head.fit(X, [e.intent for e in examples])
        self.need_head.fit(X, [e.need for e in examples])
        self.intensity_head.fit(X, [e.intensity for e in examples])
        self.social_head.fit(X, [[e.openness, e.trust, e.irritation] for e in examples])
        return self

    def predict(self, text: str) -> dict:
        X = self.vectorizer.transform([text])

        emotion_probs = self.emotion_head.predict_proba(X)[0]
        emotion_classes = self.emotion_head.classes_
        # top-3 emotions as a dict, matching EQState.emotion shape
        top_idx = np.argsort(emotion_probs)[::-1][:3]
        emotion = {emotion_classes[i]: round(float(emotion_probs[i]), 3) for i in top_idx if emotion_probs[i] > 0.05}

        intent_probs = self.intent_head.predict_proba(X)[0]
        intent = self.intent_head.classes_[int(np.argmax(intent_probs))]
        intent_confidence = float(np.max(intent_probs))

        need_probs = self.need_head.predict_proba(X)[0]
        need = self.need_head.classes_[int(np.argmax(need_probs))]

        intensity = float(np.clip(self.intensity_head.predict(X)[0], 0.0, 1.0))
        social_raw = self.social_head.predict(X)[0]
        openness, trust, irritation = [float(np.clip(v, 0.0, 1.0)) for v in social_raw]

        return {
            "emotion": emotion,
            "intensity": round(intensity, 3),
            "intent": intent,
            "need": need,
            "social_state": {
                "openness": round(openness, 3),
                "trust": round(trust, 3),
                "irritation": round(irritation, 3),
            },
            "confidence": round(intent_confidence, 3),
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "PersonaEQModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def evaluate(model: PersonaEQModel, examples: list[Example]) -> EvalMetrics:
    texts = [e.text for e in examples]
    X = model.vectorizer.transform(texts)

    emotion_pred = model.emotion_head.predict(X)
    intent_pred = model.intent_head.predict(X)
    need_pred = model.need_head.predict(X)
    intensity_pred = model.intensity_head.predict(X)
    social_pred = model.social_head.predict(X)

    emotion_true = np.array([e.emotion for e in examples])
    intent_true = np.array([e.intent for e in examples])
    need_true = np.array([e.need for e in examples])
    intensity_true = np.array([e.intensity for e in examples])
    social_true = np.array([[e.openness, e.trust, e.irritation] for e in examples])

    return EvalMetrics(
        emotion_accuracy=float(np.mean(emotion_pred == emotion_true)),
        intent_accuracy=float(np.mean(intent_pred == intent_true)),
        need_accuracy=float(np.mean(need_pred == need_true)),
        intensity_mae=float(np.mean(np.abs(intensity_pred - intensity_true))),
        social_state_mae=float(np.mean(np.abs(social_pred - social_true))),
    )


def train_and_evaluate(n_per_class: int = 60, test_size: float = 0.2, seed: int = 42) -> tuple[PersonaEQModel, EvalMetrics]:
    examples = generate(n_per_class=n_per_class)
    train, test = train_test_split(examples, test_size=test_size, random_state=seed)
    model = PersonaEQModel().fit(train)
    metrics = evaluate(model, test)
    return model, metrics
