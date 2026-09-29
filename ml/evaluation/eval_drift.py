"""
Evaluation for user_model/drift.py (design doc V4, "personality
drift") — does drift detection catch real, durable preference changes
while *not* firing on the several things that superficially resemble
them?

The false-positive cases here matter more than the true-positive ones,
and every one of them is a scenario that actually broke a version of
this detector during development rather than a hypothetical:

  - `pure_decay_from_default`: a trait the person never exhibits at
    all (no emoji, ever) decays monotonically from its 0.5 default
    toward 0. Large, clean, settled change — and completely
    meaningless, because it's the estimate converging toward a truth it
    never knew, not a person who changed. The first version of this
    detector reported it as drift.
  - `erratic`: someone genuinely inconsistent. Huge apparent range, so
    any naive early-vs-recent comparison flags it constantly, but
    nothing durable actually shifted.
  - `flat` / `tiny_wobble`: the ordinary case. Most traits for most
    users never drift, and a detector that can't stay silent here is
    useless in production.

And on the true-positive side, `real_reversal_crossing_default` is the
pitch's own example (detailed -> concise) with its real-world shape:
the estimate climbs away from 0.5 before reversing, which means the
"early" window catches it mid-climb and the naive comparison nearly
misses it. That case is why peak/trough comparison exists.

Run from ml/: python3 evaluation/eval_drift.py
"""
import datetime
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.database import Base, SessionLocal, TraitHistory, User, engine  # noqa: E402
from app.user_model.drift import detect_drift  # noqa: E402

# (trait_name, values, should_detect, why)
CASES = [
    (
        "real_reversal_crossing_default",
        # The pitch's example, real captured shape: rises from the 0.5
        # default, then durably reverses well below it.
        [0.511, 0.521, 0.531, 0.54, 0.549, 0.557, 0.564, 0.572, 0.578, 0.585,
         0.591, 0.596, 0.562, 0.531, 0.501, 0.473, 0.446, 0.422, 0.398, 0.376,
         0.356, 0.336, 0.318, 0.301],
        True,
        "genuine detailed->concise reversal; early window catches it mid-climb",
    ),
    (
        "real_step_change",
        [0.5] * 8 + [0.8] * 10 + [0.7, 0.6, 0.5, 0.4] + [0.3] * 10,
        True,
        "clean sustained shift with a settled endpoint",
    ),
    (
        "pure_decay_from_default",
        # Person never exhibits this trait at all — estimate just decays.
        [0.463, 0.428, 0.396, 0.366, 0.339, 0.313, 0.29, 0.268, 0.248, 0.229,
         0.212, 0.196, 0.181, 0.168, 0.155, 0.144, 0.133, 0.123, 0.114, 0.106,
         0.098, 0.091, 0.084, 0.078],
        False,
        "one-sided convergence from the 0.5 default, not a change in the person",
    ),
    (
        "erratic",
        [0.5] * 8 + [0.2, 0.9, 0.1, 0.8, 0.3, 0.7, 0.2, 0.9] * 3,
        False,
        "inconsistent, never settles — recent-window stddev bar rejects this",
    ),
    (
        "flat",
        [0.5] * 32,
        False,
        "no change at all — the common case; detector must stay silent",
    ),
    (
        "tiny_wobble",
        [0.5, 0.52, 0.49, 0.51, 0.5, 0.48, 0.52, 0.5] * 4,
        False,
        "ordinary EWMA jitter, below the magnitude bar",
    ),
]


def build_history(db, user_id: str, trait_name: str, values: list[float]):
    t0 = datetime.datetime(2026, 1, 1)
    for i, v in enumerate(values):
        db.add(TraitHistory(
            user_id=user_id, name=trait_name, value=v, confidence=0.8,
            evidence_count=i + 1, recorded_at=t0 + datetime.timedelta(minutes=i),
        ))
    db.commit()


def main():
    Base.metadata.create_all(engine)

    correct = 0
    results = []
    print(f"{'Case':<36}{'Expected':<12}{'Got':<12}{'Status'}")
    print("-" * 72)

    for trait_name, values, should_detect, why in CASES:
        # Each case gets its own user so traits can't interfere.
        db = SessionLocal()
        user_id = f"u_{trait_name}"
        db.add(User(id=user_id))
        db.commit()
        build_history(db, user_id, trait_name, values)

        findings = detect_drift(db, user_id)
        detected = len(findings) > 0
        ok = detected == should_detect
        correct += ok
        results.append((trait_name, should_detect, detected, ok, why,
                        findings[0].describe() if findings else None))

        print(f"{trait_name:<36}{'drift' if should_detect else 'silent':<12}"
              f"{'drift' if detected else 'silent':<12}{'OK' if ok else 'WRONG'}")
        db.close()

    accuracy = correct / len(CASES)
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Correct: {correct}/{len(CASES)} = {accuracy:.1%}")

    report_path = Path(__file__).resolve().parent.parent / "models" / "drift_eval.txt"
    with open(report_path, "w") as f:
        f.write(f"Drift detection evaluation (hand-constructed cases, n={len(CASES)})\n\n")
        f.write(f"Correct: {correct}/{len(CASES)} = {accuracy:.1%}\n\n")
        for trait_name, should, got, ok, why, summary in results:
            f.write(f"{trait_name}\n")
            f.write(f"  expected={'drift' if should else 'silent'} got={'drift' if got else 'silent'} [{'OK' if ok else 'WRONG'}]\n")
            f.write(f"  rationale: {why}\n")
            if summary:
                f.write(f"  reported: {summary}\n")
            f.write("\n")
        f.write(
            "The four should-stay-silent cases are the point of this evaluation.\n"
            "Three of them (pure_decay_from_default, erratic, and the naive-comparison\n"
            "failure that real_reversal_crossing_default guards against) each broke a\n"
            "version of this detector during development — this file is the regression\n"
            "test that keeps them fixed. A drift detector that fires readily is worse\n"
            "than none at all: it would tell a user their personality changed every\n"
            "time an estimate settled, which is both wrong and the kind of wrong that\n"
            "erodes trust in everything else the system claims to know about them.\n\n"
            "Caveat, same as the rest of this harness: these are hand-constructed\n"
            "series chosen to represent shapes seen during real testing, not sampled\n"
            "from production usage. They validate the detector's logic, not that these\n"
            "are the only shapes real users produce.\n"
        )

    print(f"\nSaved report to {report_path}")
    return 0 if correct == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
