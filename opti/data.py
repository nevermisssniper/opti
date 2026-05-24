"""Synthetic daily log generation with known ground-truth structure.

Why synthetic? Lets us validate that the full pipeline (features -> model ->
optimizer) recovers known coefficients within tolerance. Real-world logging
gives you ~30 noisy rows; synthetic gives you a known answer to test against.

Behavioral inputs (things the user controls or measures daily):
    sleep_hours, hrv_ms, training_load, study_minutes, screen_time,
    caffeine_mg, steps, prior_review_count

Performance outputs (next-day, what we want to maximize):
    mental_score:    focus/mood composite, 0-100
    physical_score:  readiness for next training session, 0-100
    cognitive_score: retention/recall of studied material, 0-100
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BEHAVIOR_COLS: list[str] = [
    "sleep_hours",
    "hrv_ms",
    "training_load",
    "study_minutes",
    "screen_time",
    "caffeine_mg",
    "steps",
    "prior_review_count",
]

PERFORMANCE_COLS: list[str] = [
    "mental_score",
    "physical_score",
    "cognitive_score",
]


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation.

    The ground-truth structure encodes three things we want the model to learn:

    1. Main effects: e.g., sleep -> mental_score (positive).
    2. Cross-domain interactions: e.g., sleep * hrv -> physical_score
       (recovery is multiplicative, not additive).
    3. Temporal lags: yesterday's training_load hurts today's physical_score
       (fatigue), two-day-old review counts help cognitive_score (spaced
       repetition).
    """

    n_days: int = 180
    noise_std: float = 3.0
    seed: int = 42

    # Bounds for each behavioral input (used by the optimizer too).
    bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "sleep_hours": (4.0, 10.0),
            "hrv_ms": (30.0, 90.0),
            "training_load": (0.0, 100.0),
            "study_minutes": (0.0, 240.0),
            "screen_time": (0.0, 480.0),
            "caffeine_mg": (0.0, 400.0),
            "steps": (1000.0, 20000.0),
            "prior_review_count": (0.0, 50.0),
        }
    )


def generate_synthetic_logs(
    cfg: SyntheticDataConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Generate a daily-log DataFrame plus the ground-truth weight dict.

    Returns
    -------
    df : pd.DataFrame
        Daily log with behavioral columns and next-day performance columns.
        Indexed by date.
    truth : dict
        {'mental_score': w_vec, 'physical_score': w_vec, 'cognitive_score': w_vec}
        for downstream recovery tests. Each w_vec is over the *featurized*
        column space (main + interactions + lags), so tests must use the same
        FeatureBuilder to compare.
    """
    cfg = cfg or SyntheticDataConfig()
    rng = np.random.default_rng(cfg.seed)

    # Sample behaviors uniformly within bounds with mild autocorrelation
    # (real human behavior is autocorrelated; uniform iid would be unrealistic
    # and would also make lagged features trivially uninformative).
    n = cfg.n_days
    behaviors = {}
    for col, (lo, hi) in cfg.bounds.items():
        # AR(1) with phi=0.4, scaled to bounds.
        raw = np.zeros(n)
        raw[0] = rng.uniform(lo, hi)
        for t in range(1, n):
            mean = 0.4 * raw[t - 1] + 0.6 * ((lo + hi) / 2)
            raw[t] = np.clip(rng.normal(mean, (hi - lo) * 0.15), lo, hi)
        behaviors[col] = raw

    df = pd.DataFrame(behaviors)
    df.index = pd.date_range("2025-01-01", periods=n, freq="D")
    df.index.name = "date"

    # Build performance scores with KNOWN structure.
    # Standardize behaviors first so coefficients are interpretable and
    # comparable in magnitude (this is also what we do at training time).
    z = (df[BEHAVIOR_COLS] - df[BEHAVIOR_COLS].mean()) / df[BEHAVIOR_COLS].std()

    # Main effects: encode the obvious priors.
    mental = (
        8.0 * z["sleep_hours"]
        + 5.0 * z["hrv_ms"]
        - 4.0 * z["screen_time"]
        + 2.0 * z["steps"]
    )
    physical = (
        6.0 * z["sleep_hours"]
        + 7.0 * z["hrv_ms"]
        - 3.0 * z["training_load"]  # same-day load hurts same-day readiness
        + 1.5 * z["steps"]
    )
    cognitive = (
        5.0 * z["sleep_hours"]
        + 4.0 * z["study_minutes"]
        + 3.0 * z["prior_review_count"]
        - 2.5 * z["screen_time"]
    )

    # Cross-domain interactions: sleep * hrv is the canonical "recovery" term.
    sleep_hrv = z["sleep_hours"] * z["hrv_ms"]
    mental = mental + 3.0 * sleep_hrv
    physical = physical + 4.5 * sleep_hrv

    # Caffeine * sleep_debt-style interaction: caffeine helps mental on
    # low-sleep days but does nothing on high-sleep days.
    mental = mental + 2.0 * z["caffeine_mg"] * (-z["sleep_hours"])

    # Lagged effects (the "spaced repetition" and "recovery debt" story).
    # Yesterday's training_load reduces today's physical (fatigue carryover).
    # Two-day-ago review boosts today's cognitive (spacing effect).
    lag1_load = z["training_load"].shift(1).fillna(0)
    lag2_review = z["prior_review_count"].shift(2).fillna(0)
    physical = physical - 3.5 * lag1_load
    cognitive = cognitive + 3.0 * lag2_review

    # Center at 60, add noise, clip to [0, 100].
    noise = rng.normal(0, cfg.noise_std, size=(n, 3))
    df["mental_score"] = np.clip(60 + mental + noise[:, 0], 0, 100)
    df["physical_score"] = np.clip(60 + physical + noise[:, 1], 0, 100)
    df["cognitive_score"] = np.clip(60 + cognitive + noise[:, 2], 0, 100)

    # We do NOT return a featurized truth-weight vector here because feature
    # ordering depends on FeatureBuilder. Tests reconstruct expected effects
    # by checking specific learned coefficients against the values above.
    truth = {
        "main_effects": {
            "mental_score": {"sleep_hours": 8.0, "hrv_ms": 5.0, "screen_time": -4.0},
            "physical_score": {"sleep_hours": 6.0, "hrv_ms": 7.0, "training_load": -3.0},
            "cognitive_score": {"study_minutes": 4.0, "prior_review_count": 3.0},
        },
        "interactions": {
            "sleep_hours_x_hrv_ms": {"mental_score": 3.0, "physical_score": 4.5},
        },
        "lags": {
            "training_load_lag1": {"physical_score": -3.5},
            "prior_review_count_lag2": {"cognitive_score": 3.0},
        },
    }
    return df, truth
