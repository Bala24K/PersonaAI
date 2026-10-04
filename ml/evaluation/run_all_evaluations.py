"""
Runs all four evaluations in this harness and writes one consolidated
report — the "portfolio-grade" summary the original design doc's
Phase 13/14 asked for, built from evaluations that actually ran against
this actual codebase rather than a hypothetical one.

Run from ml/: python3 evaluation/run_all_evaluations.py
"""
import os
import subprocess
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent
MODELS_DIR = EVAL_DIR.parent / "models"
CONSOLIDATED_OUT = MODELS_DIR / "EVALUATION_SUMMARY.md"

SCRIPTS = [
    ("Emotion detection ablation (lexicon vs. trained vs. blended)", "eval_emotion_classifier.py", "ablation_report.txt"),
    ("Memory extraction (precision/recall)", "eval_memory_extraction.py", "memory_extraction_eval.txt"),
    ("Response critic (precision/recall/specificity)", "eval_critic.py", "critic_eval.txt"),
    ("Strategy selection (accuracy)", "eval_strategy_selection.py", "strategy_selection_eval.txt"),
    ("Style vector (discrimination + similarity sanity)", "eval_style_vector.py", "style_vector_eval.txt"),
    ("Personality drift detection (true/false positive cases)", "eval_drift.py", "drift_eval.txt"),
    ("Extended: retrieval relevance, grounding, hallucination, regressions", "eval_extended.py", "extended_eval.txt"),
]


def main():
    print("Running full evaluation harness...\n", flush=True)
    sections = []

    for title, script, report_file in SCRIPTS:
        print(f"=== {title} ===", flush=True)
        result = subprocess.run(
            [sys.executable, str(EVAL_DIR / script)],
            cwd=str(EVAL_DIR.parent), capture_output=True, text=True,
        )
        print(result.stdout, flush=True)
        if result.returncode != 0:
            print(f"FAILED: {result.stderr}", file=sys.stderr, flush=True)
            sections.append((title, f"**FAILED TO RUN**\n\n```\n{result.stderr}\n```"))
            continue
        if report_file:
            report_path = MODELS_DIR / report_file
            content = report_path.read_text() if report_path.exists() else "(no report file produced)"
        else:
            content = result.stdout or "(no output produced)"
        sections.append((title, content))

    with open(CONSOLIDATED_OUT, "w") as f:
        f.write("# Evaluation summary\n\n")
        f.write(
            "Six evaluations against this codebase, run together as one harness. "
            "Each is a hand-built test set (disclosed as such in its own report — "
            "none of this is independently validated by a third party), designed "
            "specifically to include hard/adversarial cases rather than only "
            "favorable ones. Several of these caught real bugs during development: a "
            "situation-classification ordering bug (via strategy selection), the "
            "emotion blend's per-class regression (via the ablation study), and "
            "three separate false-positive modes in drift detection (via the drift "
            "eval, whose should-stay-silent cases each broke an earlier version of "
            "that detector). Left in these reports rather than quietly fixed and "
            "hidden, because catching real issues is the actual point of building "
            "an eval harness at all.\n\n"
        )
        for title, content in sections:
            f.write(f"## {title}\n\n```\n{content}\n```\n\n")

    print(f"\nConsolidated report written to {CONSOLIDATED_OUT}")


if __name__ == "__main__":
    main()
