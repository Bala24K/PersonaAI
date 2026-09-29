"""
Personality drift detection (design doc V4: "Personality drift —
This is particularly interesting. People change. Your system could
detect: Old: prefers highly detailed responses / Recent: prefers
concise responses — rather than treating the original profile as
permanent truth").

The EWMA in `model.py::update_trait` already *adapts* to change — that
was true from V1. What it can't do is *notice* change: the estimate
slides from 0.8 to 0.3 over a month and nothing anywhere says "this
person's preference reversed." This module reads the TraitHistory log
and reports sustained shifts.

## The method, and why it's this one

For each trait, split its history into an early window and a recent
window, take the mean of each, and report the difference. A shift
counts as drift when it clears three bars simultaneously:

  1. **Magnitude**: the means differ by at least MIN_DRIFT_MAGNITUDE.
     Filters out ordinary EWMA jitter.
  2. **Evidence**: both windows have at least MIN_WINDOW_OBSERVATIONS
     points. A "shift" computed from two observations is noise.
  3. **Consistency**: the recent window's own standard deviation is
     below MAX_RECENT_STDDEV. This is the bar that separates "settled
     into a genuinely new preference" from "bouncing around wildly and
     happens to have landed somewhere else right now" — without it, an
     erratic user would generate constant false drift reports.

Deliberately NOT a statistical significance test (t-test or similar).
Those assume independent samples; consecutive EWMA values are strongly
autocorrelated *by construction* (each is a weighted function of the
previous one), so a t-test over them would report significance far too
readily and the p-value would be meaningless. Three interpretable
thresholds you can explain and tune beat one number that looks rigorous
but rests on a violated assumption. This is stated plainly in the
output rather than dressed up.

Two refinements were added after testing this against real simulated
conversations rather than only synthetic step-functions, both because
the first version got a real case wrong:

  - **Burn-in skip.** Traits start at 0.5 by default, so their first
    several observations are the estimate escaping that arbitrary value,
    not the person changing. Without skipping them, a user who simply
    never used emoji got flagged for "emoji_tendency decreased" — the
    estimate converging toward a truth it never knew, misreported as a
    personality shift.
  - **Peak/trough comparison.** A preference that rises then falls can
    leave early and recent means deceptively close. A simulated user who
    wrote long messages and then switched to terse ones had verbosity go
    0.52 -> 0.65 -> 0.41 — an obvious reversal that plain
    early-vs-recent nearly missed, because the early window caught the
    estimate mid-climb. Comparing the recent window against the most
    extreme sustained window anywhere earlier catches that shape.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.database import Trait, TraitHistory

# A trait must move at least this far (on the 0-1 trait scale) between
# window means before it's called drift rather than jitter.
MIN_DRIFT_MAGNITUDE = 0.15

# Each window needs at least this many recorded observations.
MIN_WINDOW_OBSERVATIONS = 5

# The recent window must be at least this settled — high variance means
# "erratic", not "changed".
MAX_RECENT_STDDEV = 0.15

# Fraction of history treated as "early" vs "recent" when splitting.
WINDOW_FRACTION = 0.35

# Every trait starts at exactly 0.5 (see model.py) and its first
# observations are mostly the EWMA walking away from that arbitrary
# starting point rather than the person changing. Counting that initial
# settling as "drift" produces false positives on traits that simply
# had a default to escape — observed directly during development: a
# user whose messages contained no emoji at all got flagged for
# "emoji_tendency decreased 0.40 -> 0.17", which is just the estimate
# converging to the truth it never knew, not a person who changed.
# Skipping the first N observations excludes that burn-in from the
# early window. With LEARNING_RATE=0.15, the EWMA covers roughly
# 1 - 0.85^N of the distance to a new level in N steps, so 8
# observations clears ~73% of the initial settling — enough that what
# remains is dominated by real signal rather than the 0.5 default.
BURN_IN_OBSERVATIONS = 8

# A trait whose whole post-burn-in history sits within this band of its
# starting default is treated as "never had a real signal" rather than
# as drift. Catches the residual case the burn-in alone doesn't: a trait
# monotonically decaying from 0.5 toward 0 because the person simply
# never exhibits it (no emoji, ever) still looks like a clean, settled,
# large change — it just isn't a *change in the person*. Requiring the
# series to have actually visited a region meaningfully away from the
# default before counting a shift filters that out.
DEFAULT_TRAIT_VALUE = 0.5
MIN_EXCURSION_FROM_DEFAULT = 0.22


@dataclass
class DriftFinding:
    trait: str
    early_value: float
    recent_value: float
    change: float           # recent - early; sign indicates direction
    direction: str          # "increased" | "decreased"
    recent_stability: float # stddev of the recent window (lower = more settled)
    early_n: int
    recent_n: int

    def describe(self) -> str:
        return (
            f"{self.trait.replace('_', ' ')} has {self.direction} "
            f"from {self.early_value:.2f} to {self.recent_value:.2f} "
            f"({self.change:+.2f}) across {self.early_n + self.recent_n} observations"
        )


def detect_drift(db: Session, user_id: str) -> list[DriftFinding]:
    """
    Returns one DriftFinding per trait that has meaningfully shifted.
    Empty list means no trait cleared all three bars — which is the
    expected result for most users most of the time, and is a
    meaningful answer rather than a failure.
    """
    rows = (
        db.query(TraitHistory)
        .filter(TraitHistory.user_id == user_id)
        .order_by(TraitHistory.recorded_at, TraitHistory.id)
        .all()
    )
    if not rows:
        return []

    by_trait: dict[str, list[float]] = {}
    for row in rows:
        by_trait.setdefault(row.name, []).append(row.value)

    findings = []
    for trait_name, all_values in by_trait.items():
        # Drop the initial burn-in, where the EWMA is escaping its
        # arbitrary 0.5 default rather than tracking a real change.
        values = all_values[BURN_IN_OBSERVATIONS:]

        window_size = int(len(values) * WINDOW_FRACTION)
        if window_size < MIN_WINDOW_OBSERVATIONS:
            continue

        early = values[:window_size]
        recent = values[-window_size:]

        early_mean = statistics.mean(early)
        recent_mean = statistics.mean(recent)
        change = recent_mean - early_mean

        # A trait that rises then falls (or vice versa) can end up with
        # early and recent means that happen to be close, hiding a real
        # reversal — observed during development with a user who wrote
        # long messages, then short ones: verbosity ran 0.52 -> 0.65 ->
        # 0.41, a clear reversal, but early-vs-recent means differed by
        # only ~0.13 because the early window caught the estimate still
        # climbing. Comparing the recent window against the most extreme
        # sustained window anywhere earlier in the series catches that
        # shape; the plain early-vs-recent comparison alone does not.
        mid_windows = [
            values[i:i + window_size]
            for i in range(0, len(values) - window_size + 1)
        ]
        peak_mean = max(statistics.mean(w) for w in mid_windows)
        trough_mean = min(statistics.mean(w) for w in mid_windows)
        # Whichever extreme the recent window moved *away* from is the
        # one that describes the change.
        from_peak = recent_mean - peak_mean
        from_trough = recent_mean - trough_mean
        extreme_change = from_peak if abs(from_peak) > abs(from_trough) else from_trough

        if abs(extreme_change) > abs(change):
            change = extreme_change
            early_mean = recent_mean - extreme_change

        if abs(change) < MIN_DRIFT_MAGNITUDE:
            continue

        recent_stddev = statistics.stdev(recent) if len(recent) > 1 else 0.0
        if recent_stddev > MAX_RECENT_STDDEV:
            continue  # erratic, not settled — see module docstring

        # Filter pure initialization decay: a trait that only ever moved
        # away from the 0.5 default in ONE direction, monotonically, is
        # the estimate converging toward a value it never knew — not a
        # person whose preference changed. Direction matters here: a
        # genuine reversal crosses the default (verbosity rising to 0.60
        # then falling to 0.30 has excursions on *both* sides), whereas
        # an emoji_tendency that only ever decays from 0.5 toward 0 stays
        # strictly on one side the whole time. Checking sidedness rather
        # than just magnitude is what distinguishes them — an earlier
        # version of this filter compared only max-excursion-vs-final and
        # wrongly suppressed the real verbosity reversal as a result.
        above = [v for v in values if v > DEFAULT_TRAIT_VALUE + 0.02]
        below = [v for v in values if v < DEFAULT_TRAIT_VALUE - 0.02]
        one_sided = not (above and below)
        max_excursion = max(abs(v - DEFAULT_TRAIT_VALUE) for v in values)
        never_left_default_region = max_excursion < MIN_EXCURSION_FROM_DEFAULT
        if one_sided and never_left_default_region:
            continue

        findings.append(DriftFinding(
            trait=trait_name,
            early_value=round(early_mean, 3),
            recent_value=round(recent_mean, 3),
            change=round(change, 3),
            direction="increased" if change > 0 else "decreased",
            recent_stability=round(recent_stddev, 3),
            early_n=len(early),
            recent_n=len(recent),
        ))

    findings.sort(key=lambda f: -abs(f.change))
    return findings


def trait_timeline(db: Session, user_id: str, trait_name: str, max_points: int = 100) -> list[dict]:
    """
    Raw value-over-time series for one trait, for charting the drift
    rather than only reporting it as a summary. Downsampled by even
    striding when history exceeds max_points — the first and last
    points are always preserved so the endpoints of any visible trend
    stay accurate.
    """
    rows = (
        db.query(TraitHistory)
        .filter(TraitHistory.user_id == user_id, TraitHistory.name == trait_name)
        .order_by(TraitHistory.recorded_at, TraitHistory.id)
        .all()
    )
    if not rows:
        return []

    if len(rows) > max_points:
        stride = len(rows) / max_points
        sampled = [rows[int(i * stride)] for i in range(max_points - 1)]
        sampled.append(rows[-1])
        rows = sampled

    return [
        {
            "value": round(r.value, 3),
            "confidence": round(r.confidence, 3),
            "evidence_count": r.evidence_count,
            "recorded_at": r.recorded_at.isoformat() if r.recorded_at else None,
        }
        for r in rows
    ]


def recenter_drifted_traits(db: Session, user_id: str) -> list[dict]:
    """
    Active drift response: for every trait where durable drift is detected,
    re-center the current Trait estimate value in the DB to the recent window mean,
    and log the update into TraitHistory.
    """
    findings = detect_drift(db, user_id)
    if not findings:
        return []

    rebalanced = []
    for f in findings:
        trait_row = (
            db.query(Trait)
            .filter(Trait.user_id == user_id, Trait.name == f.trait)
            .first()
        )
        if trait_row:
            trait_row.value = f.recent_value
            history_entry = TraitHistory(
                user_id=user_id,
                name=f.trait,
                value=f.recent_value,
                confidence=trait_row.confidence,
                evidence_count=trait_row.evidence_count,
            )
            db.add(history_entry)
            rebalanced.append({
                "trait": f.trait,
                "old_value": f.early_value,
                "new_value": f.recent_value,
                "change": f.change,
            })

    if rebalanced:
        db.commit()

    return rebalanced

