"""
Three-layer emotional-intelligence pipeline (design doc Phase 5 / 17):

    Layer 1 — Emotion:    what is the person feeling
    Layer 2 — Situation:  what kind of thing is happening to them
    Layer 3 — Strategy:   what should the response actually try to do

Kept as three separate, inspectable functions rather than one opaque
call, exactly per the design doc's reasoning: it's more interesting (and
more debuggable) than a single sentiment score, and each layer can be
independently upgraded later (e.g. swap Layer 1 for a trained classifier)
without touching the other two.

Terminology note (kept consistent everywhere in this codebase and in
responses that reference it): this pipeline is described as
"emotion-aware" / "EQ-oriented" pattern matching over observable text,
not as the system "having" emotional intelligence. It infers latent
signals from language; it doesn't perceive feelings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.emotion.lexicon import EMOTION_LEXICON, INTENSIFIERS, NEGATORS, FLAT_AFFECT_MARKERS
from app.emotion import trained_classifier

# How much weight the trained classifier vs. the lexicon gets when both
# fire. The trained model is evaluated (see trained/training_report.txt:
# 40.2% accuracy / 0.333 macro F1 on 6 classes) and generalizes beyond
# exact word matches; the lexicon is fully interpretable, handles
# negation explicitly, and is the only signal for fear/disgust/frustration
# (absent from the training data — see trained_classifier.py docstring).
TRAINED_WEIGHT = 0.65
LEXICON_WEIGHT = 0.35

SITUATION_PATTERNS = {
    "rejection": [r"\breject(ed|ion)?\b", r"\bturned down\b", r"\bghosted\b", r"\bdidn'?t (get|make it)\b"],
    "failure": [r"\bfail(ed|ing)?\b", r"\bmessed up\b", r"\bscrewed up\b", r"\bblew it\b"],
    "conflict": [r"\bfight\b", r"\bargument\b", r"\bmad at me\b", r"\byelled\b", r"\bbroke up\b"],
    # "loss" is checked before "achievement" deliberately: "passed away"
    # would otherwise match achievement's bare \bpassed\b pattern first
    # and misclassify a death as good news (a real bug this ordering
    # fixes — caught by ml/evaluation/eval_strategy_selection.py, not
    # spotted in ad hoc testing). The negative lookahead on "passed" is a
    # second, independent safeguard against the same failure mode, so
    # this stays correct even if these dicts get reordered again later.
    "loss": [r"\bpassed away\b", r"\blost (my|him|her|them)\b", r"\bdied\b"],
    "achievement": [r"\bgot the (job|offer)\b", r"\bpassed\b(?!\s+away)", r"\bnailed it\b", r"\bproud of\b", r"\bwon\b"],
    "uncertainty": [r"\bnot sure\b", r"\bdon'?t know what to do\b", r"\bshould i\b", r"\bwhat if\b"],
    "social": [r"\bfriend\b", r"\bparty\b", r"\bhang out\b", r"\bmeet up\b"],
    "overload": [r"\bso much (to do|work)\b", r"\bdeadline\b", r"\bburn(ed|t) out\b", r"\bcan'?t keep up\b"],
}

# (emotion, situation) -> strategy. `default` is used when no specific
# combination matches. This table is Phase 17's
# "Emotion + Situation + Preference -> Response strategy" made explicit
# and editable, rather than left implicit inside an LLM prompt.
STRATEGY_TABLE: dict[tuple[str, str], str] = {
    ("sadness", "rejection"): "acknowledge_then_offer_choice",
    ("frustration", "rejection"): "acknowledge_then_offer_choice",
    ("sadness", "failure"): "acknowledge_then_offer_choice",
    ("anger", "conflict"): "listen_first",
    ("sadness", "loss"): "listen_first",
    ("joy", "achievement"): "celebrate",
    ("anxiety", "overload"): "help_prioritize",
    ("anxiety", "uncertainty"): "ask_clarifying_question",
    ("fear", "uncertainty"): "ask_clarifying_question",
}
# Fallback when the situation is clearly identifiable but the lexicon
# didn't confidently detect an emotion (e.g. "wtf", "fucking" signal
# frustration to a human reader but aren't themselves emotion words).
# Situation alone is often informative enough to pick a sane default —
# this is what stops "I got rejected again" from ever landing on the
# generic answer_directly strategy just because no emotion word fired.
SITUATION_DEFAULT_STRATEGY = {
    "rejection": "acknowledge_then_offer_choice",
    "failure": "acknowledge_then_offer_choice",
    "conflict": "listen_first",
    "loss": "listen_first",
    "achievement": "celebrate",
    "overload": "help_prioritize",
    "uncertainty": "ask_clarifying_question",
}

DEFAULT_STRATEGY_BY_EMOTION = {
    "sadness": "listen_first",
    "anger": "listen_first",
    "frustration": "acknowledge_then_offer_choice",
    "anxiety": "help_prioritize",
    "fear": "ask_clarifying_question",
    "joy": "celebrate",
    "surprise": "ask_clarifying_question",
    "disgust": "listen_first",
    "neutral": "answer_directly",
}


@dataclass
class EmotionResult:
    scores: dict[str, float]         # emotion -> probability-like score, sums roughly to 1
    dominant: str
    situation: str
    strategy: str
    flat_affect_flag: bool = False   # "I'm fine" energy — literal words undercut by context
    confidence: float = 0.0
    trained_model_used: bool = False  # whether the trained classifier contributed to `scores`


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _detect_emotions(text: str) -> dict[str, float]:
    words = _tokenize(text)
    bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
    raw_scores: dict[str, float] = {e: 0.0 for e in EMOTION_LEXICON}

    def _score_hit(emotion: str, i: int):
        weight = 1.0
        window = words[max(0, i - 2):i]
        if any(neg in window for neg in NEGATORS):
            # crude negation handling: "not happy" shouldn't add to joy
            weight *= -0.6
        if any(intens in window for intens in INTENSIFIERS):
            weight *= 1.5
        raw_scores[emotion] += max(weight, 0)

    for i, w in enumerate(words):
        for emotion, lex in EMOTION_LEXICON.items():
            if w in lex:
                _score_hit(emotion, i)

    # Multi-word lexicon entries (e.g. "sick of", "tired of") can never
    # match against single tokens above — check bigrams too. Indexed by
    # the position of the bigram's *first* word so the negation/intensifier
    # window lines up the same way.
    for i, bg in enumerate(bigrams):
        for emotion, lex in EMOTION_LEXICON.items():
            if bg in lex:
                _score_hit(emotion, i)

    total = sum(raw_scores.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw_scores.items() if v > 0}


def _detect_flat_affect(text: str, lexical_scores: dict[str, float]) -> bool:
    """
    "Yeah I'm fine lol." — literal words read neutral/positive, but the
    combination of a flat-affect marker with a dismissive tag (lol, whatever,
    a short clipped reply) is a classic mask for distress. This nudges the
    system to not take "fine" at face value, per design doc section 16.
    """
    lowered = text.lower().strip()
    has_marker = any(re.search(rf"\b{m}\b", lowered) for m in FLAT_AFFECT_MARKERS)
    dismissive_tail = bool(re.search(r"\b(lol|lmao|whatever|i guess)\b\.?$", lowered))
    very_short = len(lowered.split()) <= 6
    return has_marker and (dismissive_tail or very_short) and not lexical_scores


def _blend_scores(trained_probs: dict[str, float], lexical_scores: dict[str, float]) -> dict[str, float]:
    """
    Combine the trained classifier's distribution (over neutral/anxiety/
    joy/sadness/surprise/anger) with the lexicon's distribution (over all
    8 categories, including fear/disgust/frustration which the trained
    model never sees). Weighted sum, renormalized. If either source is
    empty (model failed to load, or no lexicon words matched), the other
    fully determines the shape — this degrades gracefully to "V1
    lexicon-only" behavior if the trained model isn't available for any
    reason.

    One deliberate asymmetry: for categories the trained model literally
    cannot predict (fear/disgust/frustration — absent from its training
    data, see trained_classifier.py), the lexicon's signal is NOT scaled
    down by LEXICON_WEIGHT the way it is for categories both sources can
    weigh in on. Diluting it there would be wrong: the trained model
    isn't *disagreeing* that a message expresses fear, it was simply
    never given fear as an option, so its confident mass sitting on
    neutral/anxiety/etc. shouldn't get to outvote a clear lexicon signal
    for a category outside its vocabulary entirely. Without this, "I am
    terrified of the exam" loses to "neutral" purely because the trained
    model has nowhere else to put its probability mass.
    """
    combined: dict[str, float] = {}
    for label, p in trained_probs.items():
        combined[label] = combined.get(label, 0.0) + TRAINED_WEIGHT * p
    for label, p in lexical_scores.items():
        if trained_probs and label not in trained_classifier.TRAINED_LABELS:
            combined[label] = combined.get(label, 0.0) + p
        else:
            combined[label] = combined.get(label, 0.0) + LEXICON_WEIGHT * p
    total = sum(combined.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in combined.items()}


def _detect_situation(text: str) -> str:
    lowered = text.lower()
    for situation, patterns in SITUATION_PATTERNS.items():
        if any(re.search(p, lowered) for p in patterns):
            return situation
    return "general"


def analyze(text: str) -> EmotionResult:
    lexical_scores = _detect_emotions(text)
    trained_probs = trained_classifier.predict(text)
    flat_affect = _detect_flat_affect(text, lexical_scores)

    if flat_affect:
        # Override: treat as mild-to-moderate frustration/sadness rather
        # than neutral, but with lower confidence since it's inferred, not
        # stated outright. Deliberately bypasses the trained model here —
        # a masked "I'm fine" is exactly the case where surface-level
        # classification (trained or not) reads wrong; that's the whole
        # point of this override existing.
        scores = {"frustration": 0.55, "sadness": 0.3, "neutral": 0.15}
        dominant = "frustration"
        confidence = 0.4
    else:
        combined = _blend_scores(trained_probs, lexical_scores)
        if combined:
            scores = combined
            dominant = max(scores, key=scores.get)
            confidence = min(0.5 + scores[dominant], 0.95)
        else:
            scores = {"neutral": 1.0}
            dominant = "neutral"
            confidence = 0.5

    situation = _detect_situation(text)
    if (dominant, situation) in STRATEGY_TABLE:
        # Most specific: an exact (emotion, situation) combination.
        strategy = STRATEGY_TABLE[(dominant, situation)]
    elif situation in SITUATION_DEFAULT_STRATEGY:
        # Next most specific: the situation itself is informative enough
        # to pick a sane strategy even without an exact emotion match.
        # This intentionally outranks the generic per-emotion default —
        # e.g. "I got rejected again" should get acknowledge_then_offer_choice
        # regardless of whether the detected dominant emotion happens to
        # be sadness, anxiety, or anger; the situation is doing the work.
        strategy = SITUATION_DEFAULT_STRATEGY[situation]
    else:
        # Least specific: no situation pattern matched (situation="general"),
        # fall back to whatever this emotion usually calls for.
        strategy = DEFAULT_STRATEGY_BY_EMOTION.get(dominant, "answer_directly")

    return EmotionResult(
        scores={k: round(v, 3) for k, v in scores.items()},
        dominant=dominant,
        situation=situation,
        strategy=strategy,
        flat_affect_flag=flat_affect,
        confidence=round(confidence, 3),
        trained_model_used=bool(trained_probs) and not flat_affect,
    )
