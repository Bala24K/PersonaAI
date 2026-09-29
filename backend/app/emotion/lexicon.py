"""
A compact, hand-built emotion lexicon.

This is the honest, documented stand-in for what the design doc calls
an "emotion model" (Phase 5). A production system would fine-tune or use
a pretrained emotion classifier (e.g. a RoBERTa emotion model) — this
sandbox can't download one, so this file plays that role for V1: a fixed
mapping from words to emotions, combined with negation/intensifier
handling in engine.py.

The important architectural point (documented at length in README) is
that emotion.engine exposes one function, `analyze(text) -> EmotionResult`.
Swapping this lexicon for a real trained classifier later means changing
this file and nothing that calls it.
"""

EMOTION_LEXICON: dict[str, list[str]] = {
    "joy": ["happy", "glad", "excited", "great", "awesome", "thrilled", "proud", "relieved", "yay", "love"],
    "sadness": ["sad", "down", "depressed", "hopeless", "lonely", "miserable", "empty", "hurt", "crying", "cry"],
    "anger": ["angry", "furious", "pissed", "mad", "annoyed", "irritated", "livid", "resentful"],
    "fear": ["scared", "afraid", "terrified", "nervous", "worried", "panicking", "dread"],
    "anxiety": ["anxious", "overwhelmed", "stressed", "stress", "panic", "tense", "on edge"],
    "frustration": ["frustrated", "stuck", "broken", "ugh", "argh", "sick of", "tired of", "over it"],
    "disgust": ["disgusted", "disgusting", "gross", "sick", "repulsed", "revolting", "nasty", "yuck"],
    "surprise": ["shocked", "surprised", "wow", "unexpected", "unbelievable"],
}

INTENSIFIERS = {"very", "extremely", "so", "really", "super", "incredibly", "totally"}
NEGATORS = {"not", "n't", "no", "never", "hardly", "barely"}

# Words that flip the literal sentiment of what follows into something
# that's usually masking distress ("fine", "okay" said flatly / with "lol"
# right after strong context) — used by engine.py's context override.
FLAT_AFFECT_MARKERS = {"fine", "okay", "ok", "whatever", "nothing"}
