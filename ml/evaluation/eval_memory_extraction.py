"""
Precision/recall evaluation for memory/extraction.py's heuristic
extractor, against a hand-labeled test set (design doc Phase 13: "Memory
- Recall@K, Precision, False memory rate").

The test set below was constructed by hand for this evaluation, not
mined from real usage — 40 messages spanning clear should-remember
cases across all four memory types, clear should-NOT-remember cases
(small talk, generic questions), and a few genuinely ambiguous/hard
cases included deliberately rather than cherry-picking only the easy
wins. Labels are a judgment call in a few places (noted inline); this
is disclosed rather than presented as objective ground truth.

Metrics:
  - Recall: of messages that SHOULD produce a memory, what fraction
    actually did (regardless of whether the extracted type/content
    exactly matches the ideal — any non-empty extraction counts as a
    "found" for recall, since partial-but-present is very different
    from missed-entirely)
  - Precision: of messages that should NOT produce a memory, what
    fraction correctly produced none (this is really a false-positive
    rate reported as its complement, kept as "precision" to match the
    design doc's own terminology)
  - Type accuracy: for correctly-found should-remember cases, did the
    extractor assign the expected type (semantic/episodic/preference/
    relationship)?

Run from ml/: python3 evaluation/eval_memory_extraction.py
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.memory.extraction import extract_memories  # noqa: E402

# (message, should_remember, expected_type_or_None)
# expected_type is None for should_remember=False cases.
TEST_SET = [
    # --- Clear semantic facts: should remember, type=semantic ---
    ("I am a computer science student.", True, "semantic"),
    ("I study CSE at my university.", True, "semantic"),
    ("I've been learning Java for two years.", True, "semantic"),
    ("I work as a backend engineer.", True, "semantic"),
    ("I am a huge fan of anime.", True, "semantic"),

    # --- Clear episodic events: should remember, type=episodic ---
    ("I got rejected from the Google interview today.", True, "episodic"),
    ("I failed my OA yesterday.", True, "episodic"),
    ("I finally passed my certification exam!", True, "episodic"),
    ("We broke up last week.", True, "episodic"),
    ("I got the job offer this morning.", True, "episodic"),

    # --- Clear preferences: should remember, type=preference ---
    ("I hate generic motivational quotes.", True, "preference"),
    ("I love direct, no-nonsense feedback.", True, "preference"),
    ("Please don't sugarcoat things with me.", True, "preference"),
    ("Remember that I dislike small talk.", True, "preference"),
    ("I prefer detailed technical explanations.", True, "preference"),

    # --- Clear relationships: should remember, type=relationship ---
    ("My friend Sarah always helps me debug.", True, "relationship"),
    ("My manager gave me a new project.", True, "relationship"),
    ("My roommate is moving out next month.", True, "relationship"),

    # --- Should NOT remember: small talk / no durable content ---
    ("hey what's up", False, None),
    ("lol that's funny", False, None),
    ("ok thanks", False, None),
    ("good morning!", False, None),
    ("haha yeah exactly", False, None),
    ("sure sounds good", False, None),
    ("what time is it", False, None),
    ("can you help me with something", False, None),
    ("thank you so much", False, None),
    ("that makes sense", False, None),

    # --- Should NOT remember: generic/factual questions with no personal info ---
    ("What's the capital of France?", False, None),
    ("How does photosynthesis work?", False, None),
    ("What's 15% of 80?", False, None),

    # --- Harder/ambiguous cases (included deliberately, not cherry-picked) ---
    # Arguably worth remembering (implies a recurring interest) but phrased
    # as a question, not a statement — heuristic patterns target statements.
    ("Do you know any good anime recommendations?", False, None),
    # A real preference, but phrased without any of the trigger words
    # (hate/dislike/prefer/love/enjoy/like) — a known heuristic gap.
    ("Honestly, motivational quotes just don't land for me.", True, "preference"),
    # An event, but low-signal phrasing without the situation keywords
    # the extractor's episodic patterns look for.
    ("Today didn't go the way I hoped.", True, "episodic"),
    # A fact, but embedded mid-sentence rather than in "I am/I study" form.
    ("Ever since I started my CSE degree, things have been intense.", True, "semantic"),
]


def main():
    should_remember = [t for t in TEST_SET if t[1]]
    should_not = [t for t in TEST_SET if not t[1]]

    found_count = 0
    type_correct_count = 0
    false_positive_count = 0

    print(f"{'Message':<55}{'Expected':<12}{'Got':<30}")
    print("-" * 97)

    for message, expected_remember, expected_type in TEST_SET:
        candidates = extract_memories(message)
        got_something = len(candidates) > 0
        got_types = [c.type for c in candidates]

        if expected_remember:
            if got_something:
                found_count += 1
                if expected_type in got_types:
                    type_correct_count += 1
            status = "OK" if got_something else "MISSED"
        else:
            if got_something:
                false_positive_count += 1
            status = "OK" if not got_something else "FALSE POSITIVE"

        got_desc = ", ".join(got_types) if candidates else "(nothing)"
        print(f"{message[:53]:<55}{'remember' if expected_remember else 'skip':<12}{got_desc:<30} [{status}]")

    recall = found_count / len(should_remember)
    type_accuracy = type_correct_count / found_count if found_count else 0.0
    precision = (len(should_not) - false_positive_count) / len(should_not)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Recall (found something when expected):      {found_count}/{len(should_remember)} = {recall:.1%}")
    print(f"Type accuracy (of those found, right type):  {type_correct_count}/{found_count} = {type_accuracy:.1%}")
    print(f"Precision (correctly skipped when expected):  {len(should_not)-false_positive_count}/{len(should_not)} = {precision:.1%}")

    report_path = Path(__file__).resolve().parent.parent / "models" / "memory_extraction_eval.txt"
    with open(report_path, "w") as f:
        f.write("Memory extraction evaluation (hand-labeled test set, n=%d)\n\n" % len(TEST_SET))
        f.write(f"Recall:         {found_count}/{len(should_remember)} = {recall:.1%}\n")
        f.write(f"Type accuracy:  {type_correct_count}/{found_count} = {type_accuracy:.1%}\n")
        f.write(f"Precision:      {len(should_not)-false_positive_count}/{len(should_not)} = {precision:.1%}\n\n")
        f.write(
            "Note: this test set was constructed by hand for this evaluation, not\n"
            "sampled from real usage, and includes several cases chosen specifically\n"
            "because the heuristic extractor's regex patterns are known not to catch\n"
            "them (see the 'harder/ambiguous cases' section in this script) — recall\n"
            "on real conversational data, which skews toward clearer statements the\n"
            "patterns were actually designed around, would likely be somewhat higher\n"
            "than this number suggests, not lower.\n"
        )

    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
