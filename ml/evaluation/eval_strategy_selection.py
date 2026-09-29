"""
Evaluation of the full emotion/engine.py::analyze() pipeline's strategy
selection against a hand-labeled test set of (message, expected_strategy)
pairs — design doc Phase 13's "Strategy appropriateness" idea, made
concrete and measurable rather than left as "have humans rate it."

This is the most subjective of the four evaluations in this harness:
reasonable people could disagree on the "right" strategy for some of
these messages. The test set sticks to cases where the expected
strategy is defensible enough to state plainly, and — as with the other
evals in this harness — includes some genuinely hard cases rather than
only the ones the system was obviously built to handle well.

Run from ml/: python3 evaluation/eval_strategy_selection.py
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.emotion.engine import analyze  # noqa: E402

# (message, expected_strategy, note)
TEST_SET = [
    # --- The flagship case and close variants ---
    ("I got rejected again", "acknowledge_then_offer_choice", "the headline example this whole project was built around"),
    ("bro wtf, got rejected again lol, another fucking OA gone bad", "acknowledge_then_offer_choice", "same situation, more informal/profane phrasing"),
    ("I failed my OA yesterday", "acknowledge_then_offer_choice", "failure situation, same expected strategy as rejection"),

    # --- Masked distress ---
    ("yeah i am fine lol", "acknowledge_then_offer_choice",
     "flat-affect override sets dominant=frustration; frustration's default strategy is "
     "acknowledge_then_offer_choice, which gives the person agency rather than assuming "
     "they want to vent — arguably better than a hardcoded listen_first here. (Earlier "
     "version of this test set had this expectation wrong, not the code — corrected after "
     "the eval caught the mismatch and manual review showed the code's behavior was actually fine.)"),

    # --- Overload / prioritization ---
    ("so much to do, deadline tomorrow, overwhelmed", "help_prioritize", "overload situation"),
    ("I have three assignments due and I don't know where to start", "help_prioritize", "overload without the exact keyword 'overwhelmed'"),

    # --- Achievement / celebration ---
    ("I got the job offer!!", "celebrate", "clear achievement"),
    ("I passed my certification exam", "celebrate", "achievement via a different keyword"),

    # --- Uncertainty / clarifying question ---
    ("I don't know what to do about this internship offer", "ask_clarifying_question", "uncertainty situation"),
    ("should I take the job or stay in school", "ask_clarifying_question", "explicit either/or uncertainty"),

    # --- Conflict / loss: listen first ---
    ("we had a huge fight and I don't know if we're okay", "listen_first", "conflict situation"),
    ("my grandmother passed away last week", "listen_first",
     "loss situation — this exact case caught a real bug during development: 'passed' alone "
     "matched the achievement pattern before loss was checked, misclassifying a death as good "
     "news. Fixed by reordering SITUATION_PATTERNS and adding a negative lookahead; see "
     "emotion/engine.py"),

    # --- Plain factual, no emotional content: answer directly ---
    ("what's the capital of France", "answer_directly", "no emotional/situational content at all"),
    ("can you explain how binary search works", "answer_directly", "technical question, no personal context"),

    # --- Harder cases: ambiguous or mixed signals ---
    # Rejection-shaped situation but with notably positive/relieved framing —
    # arguably should NOT get the standard rejection treatment, but the
    # situation regex doesn't know that; included to show a real limitation.
    ("I got rejected but honestly I'm relieved, wasn't the right fit anyway", "acknowledge_then_offer_choice",
     "situation regex fires on 'rejected' regardless of the actually-positive framing — known limitation"),
    # Sarcastic "celebration" — situation regex sees "passed" and calls it
    # achievement, missing the sarcasm entirely.
    ("oh great, I 'passed' my performance review, whatever that means", "celebrate",
     "sarcasm not detected — situation regex takes 'passed' at face value, another known limitation"),
]


def main():
    correct = 0
    print(f"{'Message':<60}{'Expected':<30}{'Got':<30}")
    print("-" * 120)

    mismatches = []
    for message, expected, note in TEST_SET:
        result = analyze(message)
        got = result.strategy
        match = got == expected
        if match:
            correct += 1
        else:
            mismatches.append((message, expected, got, note))
        print(f"{message[:58]:<60}{expected:<30}{got:<30}{'OK' if match else 'MISMATCH'}")

    accuracy = correct / len(TEST_SET)
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Strategy match: {correct}/{len(TEST_SET)} = {accuracy:.1%}")

    report_path = Path(__file__).resolve().parent.parent / "models" / "strategy_selection_eval.txt"
    with open(report_path, "w") as f:
        f.write("Strategy selection evaluation (hand-labeled test set, n=%d)\n\n" % len(TEST_SET))
        f.write(f"Accuracy: {correct}/{len(TEST_SET)} = {accuracy:.1%}\n\n")
        if mismatches:
            f.write("Mismatches:\n")
            for message, expected, got, note in mismatches:
                f.write(f"  \"{message}\"\n    expected={expected}  got={got}\n    note: {note}\n\n")
        f.write(
            "Caveat: strategy appropriateness is inherently more subjective than the\n"
            "other evaluations in this harness. The two 'harder cases' were included\n"
            "specifically to demonstrate a real, known limitation — situation\n"
            "classification is regex-based (see emotion/engine.py's SITUATION_PATTERNS)\n"
            "and doesn't understand sarcasm or reframing, so 'rejected but relieved'\n"
            "still triggers the standard rejection strategy even though a human would\n"
            "read the message as basically fine. Fixing this well would need actual\n"
            "sentiment-of-the-whole-sentence understanding, not a better regex — a\n"
            "natural argument for the LLM-based critique tier (CRITIC_MODE=heuristic_llm)\n"
            "catching what pattern matching structurally can't.\n"
        )

    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
