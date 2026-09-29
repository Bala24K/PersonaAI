#!/usr/bin/env python3
"""Trains the EQ model on bootstrapped data and saves a real model artifact to
models/eq_model.pkl. Run: python scripts/train_eq_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.eq_model import train_and_evaluate


def main():
    print("Generating bootstrapped training data...")
    model, metrics = train_and_evaluate(n_per_class=80)

    print("\n=== Held-out evaluation ===")
    print(f"Emotion accuracy:      {metrics.emotion_accuracy:.3f}")
    print(f"Intent accuracy:       {metrics.intent_accuracy:.3f}")
    print(f"Need accuracy:         {metrics.need_accuracy:.3f}")
    print(f"Intensity MAE:         {metrics.intensity_mae:.3f}")
    print(f"Social-state MAE:      {metrics.social_state_mae:.3f}")

    out_path = "models/eq_model.pkl"
    model.save(out_path)
    print(f"\nSaved trained model to {out_path}")

    # quick manual sanity check
    print("\n=== Sanity check on unseen phrasing ===")
    samples = [
        "I am absolutely furious about how the deadline got moved without telling me",
        "just wanted to say thank you, that really helped today",
        "not sure what to do, kind of worried about how the interview went",
    ]
    for s in samples:
        print(f"  '{s}'\n    -> {model.predict(s)}")


if __name__ == "__main__":
    main()
