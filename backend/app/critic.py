"""
Response critic (design doc Phase 12: "self-correction / reflection").

Before this module, the pipeline generated one reply and sent it —
there was no check that the reply actually did what the chosen strategy
called for. A strategy of "listen_first" doesn't mean anything if
nothing checks whether the reply actually listened first.

Two tiers, matching the pattern used elsewhere in this codebase
(heuristic always available, LLM-based opt-in when a key exists):

- `heuristic_critique()`: pure text/pattern checks, zero cost, always
  runs. Catches concrete, checkable mismatches — jumping straight to
  advice when the strategy calls for acknowledgment first, a "celebrate"
  reply that opens with an apology, a cliche phrase slipping through
  despite high cliche_aversion, a reply too short for a serious
  situation. Deliberately narrow: each check is something you could
  point to in the text and defend, not a vibes-based quality score.

- `llm_critique()`: optional, requires ANTHROPIC_API_KEY. Asks the model
  directly whether the reply fits the strategy and emotional context,
  and if not, why. This can catch things the heuristics structurally
  can't (tone that's technically acknowledgment-shaped but still reads
  dismissive, for instance) — but it costs a real API call, so it's
  opt-in via config.CRITIC_MODE, not run by default.

Regeneration (one attempt, never more, to bound cost and latency) only
happens when a real model is available to act on the critique feedback
— see agents/orchestrator.py. The offline template responder can't
usefully "fix" itself based on a critique (it has no reasoning to
revise), so in offline mode the critic still runs and reports issues —
useful for demoing that it works, and honest about what it can't
actually correct — but never triggers a regeneration attempt.
"""
from __future__ import annotations

import json
import re

ADVICE_MARKERS = [
    r"\byou should\b", r"\byou need to\b", r"\bhave you tried\b",
    r"\bwhy don'?t you\b", r"\btry to\b", r"\bmaybe you could\b",
    r"\bi'?d suggest\b", r"\bi recommend\b", r"\bwhat you (should|need to) do\b",
]
ACKNOWLEDGMENT_MARKERS = [
    r"\bsorry\b", r"\bthat sounds\b", r"\bthat'?s (rough|frustrating|hard|tough|a lot)\b",
    r"\bi hear you\b", r"\bi understand\b", r"\bugh\b", r"\bdamn\b", r"\bthat stings\b",
]
NEGATIVE_OPENERS = [r"^(unfortunately|but|however|sorry)\b"]

# Same list personalize.py strips — checked again here as a second,
# independent safety net rather than trusting the first pass caught
# everything.
CLICHES = [
    "everything happens for a reason", "it is what it is", "stay positive",
    "good things take time", "when one door closes, another opens",
]

STRATEGIES_REQUIRING_ACKNOWLEDGMENT_FIRST = {"listen_first", "acknowledge_then_offer_choice"}
SERIOUS_SITUATIONS = {"rejection", "failure", "loss", "conflict"}

ISSUE_DESCRIPTIONS = {
    "skipped_acknowledgment_jumped_to_advice": (
        "The reply jumped straight to advice without first acknowledging "
        "how the person feels. Revise so it briefly acknowledges what "
        "they said before any suggestion."
    ),
    "celebration_undercut_by_negative_opener": (
        "The reply is supposed to celebrate something positive but opens "
        "on a negative or lukewarm note. Revise to open with genuine "
        "enthusiasm."
    ),
    "cliche_present_despite_high_aversion": (
        "The reply uses a generic/cliche phrase this person has said "
        "they dislike. Revise to remove it and say the same thing in a "
        "more specific, genuine way."
    ),
    "reply_too_short_for_serious_situation": (
        "The reply is too short/dismissive for how serious this "
        "situation is. Revise to give it appropriate weight."
    ),
    "style_mismatch_detected": (
        "The reply's writing style strongly diverges from the user's "
        "communication style vector. Revise to align tone, brevity, and formality."
    ),
    "empty_reply": "The reply was empty. Generate an actual response.",
}


def describe_issues(issues: list[str]) -> str:
    """Turns issue codes into feedback text for a regeneration prompt."""
    return " ".join(ISSUE_DESCRIPTIONS.get(code, code) for code in issues)


def heuristic_critique(reply: str, strategy: str, situation: str, trait_estimates: dict, style_similarity: float | None = None) -> list[str]:
    """Returns a list of issue codes (empty list = no issues found)."""
    if not reply or not reply.strip():
        return ["empty_reply"]

    lowered = reply.lower().strip()
    issues = []

    if strategy in STRATEGIES_REQUIRING_ACKNOWLEDGMENT_FIRST:
        has_advice = any(re.search(p, lowered) for p in ADVICE_MARKERS)
        has_ack = any(re.search(p, lowered) for p in ACKNOWLEDGMENT_MARKERS)
        if has_advice and not has_ack:
            issues.append("skipped_acknowledgment_jumped_to_advice")

    if strategy == "celebrate" and any(re.search(p, lowered) for p in NEGATIVE_OPENERS):
        issues.append("celebration_undercut_by_negative_opener")

    cliche_aversion = trait_estimates.get("cliche_aversion")
    if cliche_aversion and hasattr(cliche_aversion, 'confidence') and hasattr(cliche_aversion, 'value'):
        if cliche_aversion.confidence >= 0.3 and cliche_aversion.value > 0.6:
            if any(phrase in lowered for phrase in CLICHES):
                issues.append("cliche_present_despite_high_aversion")

    if situation in SERIOUS_SITUATIONS and len(lowered.split()) < 4:
        issues.append("reply_too_short_for_serious_situation")

    if style_similarity is not None and style_similarity < 0.45:
        issues.append("style_mismatch_detected")

    return issues


CRITIC_SYSTEM_PROMPT = """You are reviewing a reply from a companion AI before it's sent, checking only whether it fits the intended response strategy and emotional context — not general quality.

Return ONLY a JSON object (no prose, no markdown fences): {"ok": true} if the reply fits, or {"ok": false, "reason": "<one short sentence>"} if it doesn't. Be strict but fair — minor stylistic choices are fine; only flag a real mismatch between what the strategy called for and what the reply actually does."""


def llm_critique(reply: str, strategy: str, situation: str, dominant_emotion: str) -> dict | None:
    """
    Returns {"ok": bool, "reason": str | None}, or None if the call
    failed / no API key / response wasn't valid JSON — None means "skip
    this check", not "critique failed", so callers should treat it the
    same as not having run this tier at all.

    The `complete` import is deferred to here (not module-level)
    specifically so that `heuristic_critique()` — used by
    ml/evaluation/eval_critic.py in an environment that only installs
    ml/requirements.txt, with no `requests` package — doesn't fail to
    import just because this *other* function in the same module needs
    it. Caught by the evaluation harness's own isolated-venv fresh-
    install test, which is exactly what that test is for.
    """
    from app.llm.client import complete
    prompt = (
        f"Strategy: {strategy}\n"
        f"Situation: {situation}\n"
        f"Detected emotion: {dominant_emotion}\n"
        f"Reply: {reply}"
    )
    raw = complete(CRITIC_SYSTEM_PROMPT, prompt, max_tokens=150)
    if raw is None:
        return None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "ok" not in parsed:
        return None
    return {"ok": bool(parsed["ok"]), "reason": parsed.get("reason")}
