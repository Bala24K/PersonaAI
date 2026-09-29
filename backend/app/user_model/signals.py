"""
Feature extraction for the user model: raw message -> {trait: (observed_value, weight)}.

This is intentionally simple, explainable arithmetic rather than a
trained classifier — it's the "style vector" idea from design doc
section 16 (average sentence length, slang, punctuation, emoji, etc.),
implemented directly instead of learned. A real V2 could replace this
with a small trained model without touching model.py, since model.py
only consumes {trait: (value, weight)} dicts.

Two kinds of signal:
  - stylistic: inferred from *how* they write (weak-ish weight)
  - explicit: the person directly states a preference (strong weight,
    e.g. "I prefer direct answers" should move `directness` a lot)
"""
from __future__ import annotations

import re

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
SLANG_WORDS = {
    "lol", "lmao", "bro", "nah", "yeah", "wtf", "tbh", "ngl", "fr", "deadass",
    "gonna", "wanna", "kinda", "sus", "vibe", "vibes", "lowkey", "highkey", "af",
}
PROFANITY_WORDS = {"fuck", "fucking", "shit", "damn", "hell", "crap"}
CLICHE_PHRASES = [
    "everything happens for a reason", "it is what it is", "stay positive",
    "good things take time", "when one door closes",
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def extract_signals(text: str) -> dict[str, tuple[float, float]]:
    """Returns {trait_name: (observed_value_0_to_1, weight)}."""
    if not text or not text.strip():
        return {}

    signals: dict[str, tuple[float, float]] = {}
    words = _tokenize(text)
    n_words = max(len(words), 1)
    lowered = text.lower()

    # --- Verbosity: proxy is message length. --------------------------
    # Longer messages from the user suggest they're comfortable with (and
    # likely expect) longer, more detailed responses back.
    length_score = min(n_words / 60, 1.0)
    signals["verbosity"] = (length_score, 0.4)

    # --- Slang tendency --------------------------------------------------
    slang_hits = sum(1 for w in words if w in SLANG_WORDS)
    slang_score = min(slang_hits / max(n_words * 0.15, 1), 1.0)
    signals["slang_tendency"] = (slang_score, 0.5)

    # --- Formality is roughly the inverse of slang + profanity + informal
    # punctuation (no caps, run-on lowercase, etc.) -----------------------
    profanity_hits = sum(1 for w in words if w in PROFANITY_WORDS)
    informal_signal = min((slang_hits + profanity_hits) / max(n_words * 0.15, 1), 1.0)
    starts_capitalized = text[:1].isupper() if text else False
    ends_with_period = text.strip().endswith(".")
    formality_score = 0.5 - 0.35 * informal_signal + 0.1 * starts_capitalized + 0.1 * ends_with_period
    signals["formality"] = (max(0.0, min(1.0, formality_score)), 0.4)

    # --- Humor: emoji/laughter markers + exclamation density -------------
    laugh_markers = len(re.findall(r"\b(lol|lmao|haha+|hehe+|💀|😂)\b", lowered))
    humor_score = min(laugh_markers / 2, 1.0)
    if humor_score > 0:
        signals["humor"] = (humor_score, 0.4)

    # --- Emoji tendency ----------------------------------------------------
    emoji_hits = len(EMOJI_RE.findall(text))
    emoji_score = min(emoji_hits / 3, 1.0)
    signals["emoji_tendency"] = (emoji_score, 0.5)

    # --- Directness: explicit statements are the strongest signal here ---
    if re.search(r"\bjust (tell|give) me\b|\bbe (blunt|direct|honest)\b|\bstraight(forward)? answer\b|\bno sugar ?coat", lowered):
        signals["directness"] = (0.9, 1.0)
    elif re.search(r"\bwhatever works\b|\bnot sure\b|\bmaybe\b", lowered):
        signals["directness"] = (0.4, 0.3)

    # --- Cliche aversion: explicit statement is a very strong signal -----
    if re.search(r"(hate|don'?t (want|like)|no) .*(clich[ée]|generic|motivational (quote|speech))", lowered):
        signals["cliche_aversion"] = (0.9, 1.0)
    if any(p in lowered for p in CLICHE_PHRASES):
        # If *they themselves* lean on cliches, that's weak evidence they
        # tolerate them more than someone who explicitly objects.
        signals["cliche_aversion"] = (0.3, 0.3)

    # --- Technical detail preference --------------------------------------
    if re.search(r"\b(explain (the )?(technical|internals|how it works)|deep dive|under the hood|in detail)\b", lowered):
        signals["technical_detail"] = (0.85, 0.8)
    elif re.search(r"\b(tldr|tl;dr|just the summary|simple (version|answer)|eli5)\b", lowered):
        signals["technical_detail"] = (0.15, 0.8)

    return signals
