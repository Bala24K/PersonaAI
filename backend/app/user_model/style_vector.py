"""
Writing-style analysis (design doc Phase 6: "Personality imitation" /
"one of the coolest parts"). The pitch's own list of features to
extract:

    Average sentence length, Vocabulary richness, Punctuation patterns,
    Emoji usage, Slang, Abbreviations, Capitalization, Question
    frequency, Response length

...and then: "Compare: Person's original messages vs AI generated
messages using linguistic/statistical similarity metrics." This module
is exactly that, built directly rather than left as a plan — the
earlier `user_model/signals.py` module computes a few of these same
underlying signals (slang, emoji, formality) but folds them straight
into single 0-1 trait nudges for the response-generation pipeline;
this module keeps each feature as its own measured number and exists
specifically to *compare two people's (or a person's and an AI's)*
styles against each other, not to drive generation.

Two entry points:
  compute_style_vector(texts)      -> StyleVector for a batch of messages
  style_similarity(a, b)           -> a single 0-1 match score

The similarity metric is a deliberately simple, defensible choice:
normalize each feature into a comparable [0, 1]-ish range using fixed,
documented bounds, then report 1 - mean(|difference|) across features.
This was chosen over cosine similarity specifically for interpretability
— "your style and the AI's style are 82% aligned, driven mostly by a
mismatch in question frequency" is a sentence you can actually explain,
where a cosine similarity on arbitrarily-scaled raw features isn't.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, fields

from app.user_model.signals import EMOJI_RE, SLANG_WORDS

ABBREVIATIONS = {
    "lol", "omg", "brb", "idk", "btw", "fyi", "asap", "rn", "imo", "tbh",
    "smh", "irl", "afaik", "nvm", "ikr",
}
# Deliberately overlaps signals.py's SLANG_WORDS in a few places (e.g.
# "tbh") — abbreviation and slang are related but not identical
# registers, and forcing a clean partition between them would be more
# arbitrary than the small overlap this leaves in place.

SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
WORD_RE = re.compile(r"[a-zA-Z']+")


@dataclass
class StyleVector:
    avg_sentence_length: float      # words per sentence
    vocabulary_richness: float      # unique words / total words (type-token ratio)
    exclamation_rate: float         # exclamation marks per 100 words
    question_frequency: float       # fraction of sentences ending in "?"
    emoji_rate: float               # emojis per 100 words
    slang_rate: float                # slang words per 100 words
    abbreviation_rate: float         # abbreviations per 100 words
    capitalization_rate: float       # fraction of sentences starting with a capital letter
    avg_message_length: float        # words per message
    sample_size: int                 # how many messages this was computed from


# Fixed normalization bounds for style_similarity — chosen from what's
# actually observable in casual English text, not fit to any dataset.
# A feature at or beyond its bound is treated as maximally different
# from a feature at the opposite bound; values are clipped, not
# extrapolated past them.
_NORM_BOUNDS = {
    "avg_sentence_length": (3.0, 25.0),
    "vocabulary_richness": (0.3, 1.0),
    "exclamation_rate": (0.0, 8.0),
    "question_frequency": (0.0, 1.0),
    "emoji_rate": (0.0, 15.0),
    "slang_rate": (0.0, 15.0),
    "abbreviation_rate": (0.0, 10.0),
    "capitalization_rate": (0.0, 1.0),
    "avg_message_length": (2.0, 60.0),
}


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def compute_style_vector(texts: list[str]) -> StyleVector:
    """
    Aggregates the style features above across a batch of messages
    (typically: all of one person's messages in a conversation, or all
    of the AI's replies in the same conversation, for comparison).
    """
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return StyleVector(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    all_words: list[str] = []
    all_sentences: list[str] = []
    message_lengths: list[int] = []
    exclamation_count = 0
    emoji_count = 0
    slang_count = 0
    abbrev_count = 0
    capitalized_sentences = 0
    question_sentences = 0

    for text in texts:
        words = WORD_RE.findall(text)
        all_words.extend(w.lower() for w in words)
        message_lengths.append(len(words))
        exclamation_count += text.count("!")
        emoji_count += len(EMOJI_RE.findall(text))

        sentences = _split_sentences(text)
        all_sentences.extend(sentences)
        for s in sentences:
            if s[:1].isupper():
                capitalized_sentences += 1
        # Question detection uses the original text's punctuation, not
        # the stripped sentence list (sentence splitting consumes the
        # "?" itself), so scan the raw text for "?" segments instead.
        question_sentences += text.count("?")

        lowered_words = [w.lower() for w in words]
        slang_count += sum(1 for w in lowered_words if w in SLANG_WORDS)
        abbrev_count += sum(1 for w in lowered_words if w in ABBREVIATIONS)

    total_words = max(len(all_words), 1)
    total_sentences = max(len(all_sentences), 1)
    unique_words = len(set(all_words))

    return StyleVector(
        avg_sentence_length=total_words / total_sentences,
        vocabulary_richness=unique_words / total_words,
        exclamation_rate=(exclamation_count / total_words) * 100,
        question_frequency=min(question_sentences / total_sentences, 1.0),
        emoji_rate=(emoji_count / total_words) * 100,
        slang_rate=(slang_count / total_words) * 100,
        abbreviation_rate=(abbrev_count / total_words) * 100,
        capitalization_rate=capitalized_sentences / total_sentences,
        avg_message_length=statistics.mean(message_lengths) if message_lengths else 0,
        sample_size=len(texts),
    )


def _normalize(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def style_similarity(a: StyleVector, b: StyleVector) -> float:
    """
    1.0 = identical style on every normalized feature, 0.0 = maximally
    different on every feature (per the fixed bounds in _NORM_BOUNDS).
    `sample_size` is excluded — it's metadata about the vector, not a
    style feature to match.
    """
    diffs = []
    for f in fields(StyleVector):
        name = f.name
        if name == "sample_size":
            continue
        bounds = _NORM_BOUNDS[name]
        na = _normalize(getattr(a, name), bounds)
        nb = _normalize(getattr(b, name), bounds)
        diffs.append(abs(na - nb))
    return 1.0 - (sum(diffs) / len(diffs))


def biggest_style_gaps(a: StyleVector, b: StyleVector, top_n: int = 3) -> list[tuple[str, float]]:
    """Returns the top_n features contributing most to a's/b's style
    mismatch, as (feature_name, normalized_gap) — the "driven mostly
    by..." explanation behind a similarity score, not just the number."""
    gaps = []
    for f in fields(StyleVector):
        name = f.name
        if name == "sample_size":
            continue
        bounds = _NORM_BOUNDS[name]
        gap = abs(_normalize(getattr(a, name), bounds) - _normalize(getattr(b, name), bounds))
        gaps.append((name, gap))
    gaps.sort(key=lambda x: -x[1])
    return gaps[:top_n]
