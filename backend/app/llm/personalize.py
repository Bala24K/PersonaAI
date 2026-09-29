"""
Personalization layer (last box before the response reaches the user in
the design doc's architecture diagram).

The system prompt already asks the model to calibrate to the user model,
but prompting is a request, not a guarantee — especially in the offline
template fallback, which doesn't "read" the prompt at all. This module
does a small amount of deterministic post-processing so trait values
visibly affect the final text even when generation itself can't be
trusted to comply:

  - low verbosity preference -> trim to the first N sentences
  - low cliche_aversion tolerance (i.e. high aversion) -> strip a short
    list of stock motivational phrases if the model produced one anyway
"""
from __future__ import annotations

import re

CLICHES = [
    "everything happens for a reason",
    "it is what it is",
    "stay positive",
    "good things take time",
    "when one door closes, another opens",
]


def apply(text: str, traits: dict) -> str:
    if not text:
        return text

    verbosity = traits.get("verbosity")
    if verbosity and verbosity.confidence >= 0.4 and verbosity.value < 0.3:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(sentences) > 3:
            text = " ".join(sentences[:3])

    cliche_aversion = traits.get("cliche_aversion")
    if cliche_aversion and cliche_aversion.confidence >= 0.3 and cliche_aversion.value > 0.6:
        lowered = text.lower()
        for phrase in CLICHES:
            if phrase in lowered:
                # crude but honest: flag rather than silently rewrite
                text += f"\n\n(Note: trimmed a stock phrase your profile says you dislike.)"
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                text = pattern.sub("", text).strip()
                break

    return text
