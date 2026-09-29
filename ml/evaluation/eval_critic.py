"""
Precision/recall evaluation for critic.py's heuristic_critique(),
against a hand-labeled test set of (reply, strategy, situation,
should_flag) cases.

This matters more than it might look: a critic that flags good replies
is actively harmful (it burns an API call on a needless regeneration
and might make a fine reply worse), so false-positive rate here is at
least as important as catch rate. Every one of the four heuristic
checks gets both a case that should trigger it and at least one
close-but-shouldn't-trigger case, specifically to stress-test for false
positives rather than only demonstrating true positives.

Run from ml/: python3 evaluation/eval_critic.py
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.critic import heuristic_critique  # noqa: E402
from app.user_model.types import TraitEstimate  # noqa: E402

HIGH_CLICHE_AVERSION = {"cliche_aversion": TraitEstimate("cliche_aversion", 0.85, 0.6, 5)}

# (reply, strategy, situation, traits, should_flag, note)
TEST_SET = [
    # --- skipped_acknowledgment_jumped_to_advice: should flag ---
    ("You should just apply to more places and try to network more.",
     "acknowledge_then_offer_choice", "rejection", {}, True, "advice with zero acknowledgment"),
    ("Have you tried reaching out to the recruiter directly?",
     "listen_first", "rejection", {}, True, "advice-only under listen_first"),

    # --- same strategy, but done right: should NOT flag ---
    ("That sounds really rough. Want to talk about what happened, or think through next steps?",
     "acknowledge_then_offer_choice", "rejection", {}, False, "acknowledges before offering choice"),
    ("Ugh, that's frustrating. I'm here if you want to vent.",
     "listen_first", "rejection", {}, False, "pure acknowledgment, no advice at all"),
    ("That's rough. Have you had a chance to process it yet?",
     "listen_first", "rejection", {}, False, "acknowledgment present alongside a gentle question"),

    # --- celebration_undercut_by_negative_opener: should flag ---
    ("Unfortunately that is pretty average news.", "celebrate", "achievement", {}, True, "negative opener on celebrate"),
    ("But I guess that's something.", "celebrate", "achievement", {}, True, "lukewarm negative opener"),

    # --- celebrate done right: should NOT flag ---
    ("That is genuinely awesome, congrats!", "celebrate", "achievement", {}, False, "clean celebration"),
    ("Yes! That's such a big deal, I'm really happy for you.", "celebrate", "achievement", {}, False, "enthusiastic, no negative opener"),

    # --- cliche_present_despite_high_aversion: should flag ---
    ("Well, everything happens for a reason, right?", "listen_first", "rejection", HIGH_CLICHE_AVERSION, True, "cliche present, aversion high"),
    ("Try to stay positive about it.", "listen_first", "rejection", HIGH_CLICHE_AVERSION, True, "different cliche, aversion high"),

    # --- cliche present but aversion NOT established: should NOT flag ---
    ("Well, everything happens for a reason, right?", "listen_first", "rejection", {}, False, "cliche present but no known aversion"),
    ("Well, everything happens for a reason, right?", "listen_first", "rejection",
     {"cliche_aversion": TraitEstimate("cliche_aversion", 0.85, 0.1, 1)}, False, "aversion value high but confidence too low"),

    # --- reply_too_short_for_serious_situation: should flag ---
    ("Ok.", "listen_first", "loss", {}, True, "one word for a loss situation"),
    ("Sorry.", "listen_first", "conflict", {}, True, "one word for a conflict"),

    # --- short but for a non-serious situation: should NOT flag ---
    ("Nice!", "celebrate", "achievement", {}, False, "short reply is fine for celebration"),
    ("Sure thing.", "answer_directly", "general", {}, False, "short reply is fine for casual/general"),

    # --- empty reply: should flag ---
    ("", "answer_directly", "general", {}, True, "empty string"),
    ("   ", "answer_directly", "general", {}, True, "whitespace only"),

    # --- normal, unremarkable replies with no strategy constraints: should NOT flag ---
    ("The capital of France is Paris.", "answer_directly", "general", {}, False, "plain factual answer"),
    ("Sure, let's break that down together.", "help_prioritize", "overload", {}, False, "on-strategy, no issues"),
    ("What's actually making you unsure here?", "ask_clarifying_question", "uncertainty", {}, False, "clarifying question, on strategy"),
]


def main():
    tp = fp = tn = fn = 0
    print(f"{'Reply':<55}{'Strategy':<28}{'Expected':<10}{'Got':<10}")
    print("-" * 103)

    for reply, strategy, situation, traits, should_flag, note in TEST_SET:
        issues = heuristic_critique(reply, strategy, situation, traits)
        flagged = len(issues) > 0

        if should_flag and flagged:
            tp += 1; status = "TP"
        elif should_flag and not flagged:
            fn += 1; status = "FN (missed)"
        elif not should_flag and flagged:
            fp += 1; status = "FP (false alarm)"
        else:
            tn += 1; status = "TN"

        reply_display = reply.strip() or "(empty)"
        print(f"{reply_display[:53]:<55}{strategy:<28}{'flag' if should_flag else 'pass':<10}{status:<10}")

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Precision (of flags raised, how many were real issues):  {precision:.1%}")
    print(f"Recall (of real issues, how many were caught):            {recall:.1%}")
    print(f"Specificity (of clean replies, how many correctly passed): {specificity:.1%}")

    report_path = Path(__file__).resolve().parent.parent / "models" / "critic_eval.txt"
    with open(report_path, "w") as f:
        f.write("Response critic evaluation (hand-labeled test set, n=%d)\n\n" % len(TEST_SET))
        f.write(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}\n")
        f.write(f"Precision: {precision:.1%}\nRecall: {recall:.1%}\nSpecificity: {specificity:.1%}\n\n")
        f.write(
            "Precision and specificity matter as much as recall here: a critic with\n"
            "high recall but low specificity flags good replies constantly, which is\n"
            "worse than no critic at all (wasted regenerations, possible degradation\n"
            "of an already-fine reply). This test set was designed with that in mind —\n"
            "every check has both a triggering case and a close-but-shouldn't-trigger\n"
            "case, not just clean positive examples.\n\n"
            "Honest caveat on a perfect score: the same person who wrote the four\n"
            "heuristic checks also wrote this test set, so a clean 100% here mostly\n"
            "confirms the checks do what they were literally written to do, not that\n"
            "they'll generalize to the full variety of real replies a live model would\n"
            "produce. Real text will hedge, combine acknowledgment and advice in the\n"
            "same clause, or use phrasing these patterns don't recognize at all — this\n"
            "eval is a regression test against known cases, not independent validation.\n"
            "The honest next step would be running the critic against real generated\n"
            "replies from a live model and having a person (ideally not the one who\n"
            "wrote the heuristics) judge agreement — not done here, no live API access\n"
            "in this build environment to generate that data.\n"
        )

    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
