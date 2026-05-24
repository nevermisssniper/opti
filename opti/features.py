"""Feature engineering: standardization, cross-domain interactions, lags.

Design notes:
- We standardize first (z-score) so interaction terms aren't dominated by
  the variable with the larger raw scale (caffeine_mg in [0, 400] vs
  hrv_ms in [30, 90]).
- Interactions are HAND-SELECTED, not exhaustive polynomial expansion.
  An exhaustive expansion of 8 features gives 28 pairs and most are
  noise. We keep the 5-6 that have a behavioral story we can defend.
- Lags use only t-1 to t-3. Beyond that, autocorrelation in real human
  data decays and the marginal information content is poor relative to
  the sample cost (each lag drops `lag` rows from training).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from opti.data import BEHAVIOR_COLS, PERFORMANCE_COLS

# Hand-selected interactions with behavioral interpretations.
DEFAULT_INTERACTIONS: list[tuple[str, str]] = [
    ("sleep_hours", "hrv_ms"),         # recovery composite
    ("training_load", "hrv_ms"),       # stress-on-resilience
    ("study_minutes", "prior_review_count"),  # new-vs-review tradeoff
    ("caffeine_mg", "sleep_hours"),    # stimulant-on-rest
    ("screen_time", "sleep_hours"),    # sleep hygiene
]

DEFAULT_LAGS: list[int] = [1, 2, 3]


@dataclass
class FeatureBuilder:
    """Standardize -> add interactions -> add lags. Stateful (fit + transform).

    Why stateful: we need to apply the SAME scaling means/stds at inference
    time. A stateless function call would silently change the feature space
    when the user-provided behavior vector has a different distribution from
    the training set (which is always, since it's one row).
    """

    interactions: list[tuple[str, str]] = field(default_factory=lambda: list(DEFAULT_INTERACTIONS))
    lags: list[int] = field(default_factory=lambda: list(DEFAULT_LAGS))
    behavior_cols: list[str] = field(default_factory=lambda: list(BEHAVIOR_COLS))

    # Fitted state
    means_: pd.Series | None = None
    stds_: pd.Series | None = None
    feature_names_: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        self.means_ = df[self.behavior_cols].mean()
        self.stds_ = df[self.behavior_cols].std().replace(0, 1.0)
        # Compute feature_names_ from a transform of the training frame.
        self.feature_names_ = list(self._build_X(df).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns the design matrix X (NOT including target columns).

        Rows with NaN from lagging are dropped. Caller should align y to the
        returned index.
        """
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("FeatureBuilder must be fit before transform.")
        X = self._build_X(df)
        X = X.dropna()
        return X

    def transform_with_target(
        self, df: pd.DataFrame, target_cols: list[str] | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (X, y) aligned on the post-dropna index."""
        target_cols = target_cols or PERFORMANCE_COLS
        X = self.transform(df)
        y = df.loc[X.index, target_cols]
        return X, y

    def transform_single(self, behavior: dict[str, float] | pd.Series) -> np.ndarray:
        """Transform a SINGLE behavior vector (no lags possible).

        Used by the optimizer. Lagged features are filled with the training-set
        mean (i.e., zero after standardization) since we are optimizing today's
        behavior given fixed historical context. The caller (BehaviorOptimizer)
        can override this if it has access to recent history.
        """
        if self.means_ is None or self.stds_ is None or self.feature_names_ is None:
            raise RuntimeError("FeatureBuilder must be fit before transform_single.")
        if isinstance(behavior, dict):
            behavior = pd.Series(behavior)
        z = (behavior[self.behavior_cols] - self.means_) / self.stds_

        out = pd.Series(0.0, index=self.feature_names_)
        # Main effects
        for c in self.behavior_cols:
            out[c] = z[c]
        # Interactions
        for a, b in self.interactions:
            name = f"{a}_x_{b}"
            if name in out.index:
                out[name] = z[a] * z[b]
        # Lags remain zero (filled with population mean post-standardization).
        return out.to_numpy()

    def transform_single_with_history(
        self,
        behavior: dict[str, float] | pd.Series,
        history: pd.DataFrame,
    ) -> np.ndarray:
        """Like transform_single but pulls real lag values from history.

        `history` must be a DataFrame of recent behavior rows (most recent
        last), with at least max(self.lags) rows.
        """
        if self.means_ is None or self.stds_ is None or self.feature_names_ is None:
            raise RuntimeError("FeatureBuilder must be fit before transform_single_with_history.")
        if isinstance(behavior, dict):
            behavior = pd.Series(behavior)
        max_lag = max(self.lags) if self.lags else 0
        if len(history) < max_lag:
            raise ValueError(f"History needs >= {max_lag} rows, got {len(history)}.")

        z_today = (behavior[self.behavior_cols] - self.means_) / self.stds_
        z_hist = (history[self.behavior_cols] - self.means_) / self.stds_

        out = pd.Series(0.0, index=self.feature_names_)
        for c in self.behavior_cols:
            out[c] = z_today[c]
        for a, b in self.interactions:
            name = f"{a}_x_{b}"
            if name in out.index:
                out[name] = z_today[a] * z_today[b]
        # Lags: lag-k means "k days ago". history.iloc[-k] is the k-th most
        # recent row (history.iloc[-1] is yesterday).
        for c in self.behavior_cols:
            for k in self.lags:
                name = f"{c}_lag{k}"
                if name in out.index:
                    out[name] = z_hist[c].iloc[-k]
        return out.to_numpy()

    # ----- internal -----

    def _build_X(self, df: pd.DataFrame) -> pd.DataFrame:
        z = (df[self.behavior_cols] - self.means_) / self.stds_
        parts: list[pd.DataFrame] = [z.copy()]

        # Interactions
        if self.interactions:
            inter = pd.DataFrame(index=df.index)
            for a, b in self.interactions:
                inter[f"{a}_x_{b}"] = z[a] * z[b]
            parts.append(inter)

        # Lags (on the standardized behaviors)
        if self.lags:
            lag_df = pd.DataFrame(index=df.index)
            for c in self.behavior_cols:
                for k in self.lags:
                    lag_df[f"{c}_lag{k}"] = z[c].shift(k)
            parts.append(lag_df)

        return pd.concat(parts, axis=1)
