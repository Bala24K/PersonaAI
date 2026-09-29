"""
Validation of user_model/style_vector.py (design doc Phase 6, "one of
the coolest parts") — does the StyleVector system actually discriminate
between styles the way it should, and does the similarity metric behave
directionally sensibly? Not a precision/recall eval like the other
three in this harness (there's no "ground truth style" to check
extracted features against) — instead, a set of hand-constructed
persona pairs with a clearly correct *directional* expectation for each
feature (e.g. "the casual persona's slang_rate should be higher than
the formal persona's"), checked automatically rather than eyeballed.

Two things are checked:
1. Per-feature discrimination: for each of several formal/casual
   persona pairs, does each feature move in the expected direction?
2. Similarity metric sanity: self-similarity is always ~1.0; two
   different personas are less similar to each other than either is to
   itself; a persona is more similar to a close variant of itself than
   to a very different one.

Run from ml/: python3 evaluation/eval_style_vector.py
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.user_model.style_vector import compute_style_vector, style_similarity  # noqa: E402

FORMAL_PERSONA = [
    "I have been considering the various options available to me.",
    "Could you please provide additional information regarding this matter?",
    "I appreciate your assistance with this issue.",
    "It would be helpful to understand the timeline for this project.",
    "Thank you for taking the time to explain the process in detail.",
]

CASUAL_PERSONA = [
    "lol idk what to do tbh",
    "omg that is so funny!! 😂😂",
    "ngl this is kinda wild rn",
    "bro wtf just happened lmao",
    "yeah nah that's not it fr",
]

# A "close variant" of the casual persona — still casual, slightly
# different specific word choices, used to check that similarity
# reflects genuine closeness rather than being an all-or-nothing signal.
CASUAL_PERSONA_VARIANT = [
    "haha idk man that's rough",
    "omg no way!! 😭",
    "ngl kinda tired rn tbh",
    "lol whatever, it's fine",
    "yeah that's wild ngl",
]

# (feature, expect_casual_higher_than_formal)
DIRECTIONAL_EXPECTATIONS = [
    ("exclamation_rate", True),
    ("emoji_rate", True),
    ("slang_rate", True),
    ("abbreviation_rate", True),
    ("avg_sentence_length", False),
    ("capitalization_rate", False),
]


def main():
    formal_vec = compute_style_vector(FORMAL_PERSONA)
    casual_vec = compute_style_vector(CASUAL_PERSONA)
    casual_variant_vec = compute_style_vector(CASUAL_PERSONA_VARIANT)

    print("=== Part 1: per-feature discrimination ===\n")
    all_correct = True
    for feature, casual_should_be_higher in DIRECTIONAL_EXPECTATIONS:
        formal_val = getattr(formal_vec, feature)
        casual_val = getattr(casual_vec, feature)
        actually_higher = casual_val > formal_val
        correct = actually_higher == casual_should_be_higher
        all_correct = all_correct and correct
        direction = "casual > formal" if casual_should_be_higher else "casual < formal"
        print(f"{feature:<22} formal={formal_val:<10.3f} casual={casual_val:<10.3f} expected({direction}): {'OK' if correct else 'WRONG'}")

    print("\n=== Part 2: similarity metric sanity ===\n")
    self_sim = style_similarity(formal_vec, formal_vec)
    cross_sim = style_similarity(formal_vec, casual_vec)
    variant_sim = style_similarity(casual_vec, casual_variant_vec)

    check_a = self_sim > 0.99
    check_b = cross_sim < self_sim
    check_c = variant_sim > cross_sim

    print(f"self-similarity (formal, formal):              {self_sim:.3f}  (expect ~1.0): {'OK' if check_a else 'WRONG'}")
    print(f"cross-similarity (formal, casual):              {cross_sim:.3f}  (expect < self-similarity): {'OK' if check_b else 'WRONG'}")
    print(f"variant-similarity (casual, casual_variant):    {variant_sim:.3f}  (expect > cross-similarity): {'OK' if check_c else 'WRONG'}")

    all_correct = all_correct and check_a and check_b and check_c

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("ALL CHECKS PASSED" if all_correct else "SOME CHECKS FAILED — see WRONG above")

    report_path = Path(__file__).resolve().parent.parent / "models" / "style_vector_eval.txt"
    with open(report_path, "w") as f:
        f.write("StyleVector discrimination + similarity-metric sanity check\n\n")
        f.write(f"All checks passed: {all_correct}\n\n")
        f.write("Per-feature discrimination (formal vs. casual hand-constructed personas):\n")
        for feature, casual_should_be_higher in DIRECTIONAL_EXPECTATIONS:
            formal_val = getattr(formal_vec, feature)
            casual_val = getattr(casual_vec, feature)
            correct = (casual_val > formal_val) == casual_should_be_higher
            f.write(f"  {feature}: formal={formal_val:.3f} casual={casual_val:.3f} [{'OK' if correct else 'WRONG'}]\n")
        f.write(f"\nself_similarity={self_sim:.3f}  cross_similarity={cross_sim:.3f}  variant_similarity={variant_sim:.3f}\n\n")
        f.write(
            "Note: this validates that the StyleVector system behaves correctly on\n"
            "clearly-differentiated hand-constructed examples, not that it perfectly\n"
            "captures 'style' in some objective sense (no such ground truth exists).\n"
            "The real-world use of this system — GET /user/{id}/style, comparing an\n"
            "actual person's messages to the AI's replies — was spot-checked manually\n"
            "during development (see docs/DEMO_SCRIPT.md) rather than formally\n"
            "evaluated here, since there's no labeled dataset of 'correct' style-match\n"
            "scores for real conversations to check against.\n"
        )

    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
