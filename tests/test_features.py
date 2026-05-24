"""Feature engineering tests."""
from __future__ import annotations

import numpy as np

from opti.data import BEHAVIOR_COLS, SyntheticDataConfig, generate_synthetic_logs
from opti.features import FeatureBuilder


def test_fit_then_transform_drops_lag_rows() -> None:
    df, _ = generate_synthetic_logs(SyntheticDataConfig(n_days=50, seed=0))
    fb = FeatureBuilder(lags=[1, 2, 3]).fit(df)
    X = fb.transform(df)
    # Lag-3 drops the first 3 rows.
    assert len(X) == len(df) - 3


def test_standardization_zero_mean_unit_std() -> None:
    df, _ = generate_synthetic_logs(SyntheticDataConfig(n_days=200, seed=0))
    fb = FeatureBuilder().fit(df)
    X = fb.transform(df)
    # Main-effect columns should be ~zero mean, ~unit std on training data.
    for c in BEHAVIOR_COLS:
        assert abs(X[c].mean()) < 0.1, f"{c} mean too far from 0"
        assert 0.85 < X[c].std() < 1.15, f"{c} std too far from 1"


def test_interaction_terms_are_products() -> None:
    df, _ = generate_synthetic_logs(SyntheticDataConfig(n_days=50, seed=0))
    fb = FeatureBuilder().fit(df)
    X = fb.transform(df)
    # sleep_hours_x_hrv_ms should equal the elementwise product of the
    # standardized parents.
    expected = X["sleep_hours"] * X["hrv_ms"]
    assert np.allclose(X["sleep_hours_x_hrv_ms"], expected)


def test_lag_features_are_shifted() -> None:
    df, _ = generate_synthetic_logs(SyntheticDataConfig(n_days=50, seed=0))
    fb = FeatureBuilder(lags=[1, 2]).fit(df)
    X = fb.transform(df)
    # X has lost the first max(lags)=2 rows. sleep_hours_lag1 at row t in X
    # should equal standardized sleep at row t-1 in the original frame.
    means, stds = fb.means_, fb.stds_
    assert means is not None and stds is not None
    z_sleep = (df["sleep_hours"] - means["sleep_hours"]) / stds["sleep_hours"]
    expected = z_sleep.shift(1).loc[X.index]
    assert np.allclose(X["sleep_hours_lag1"].to_numpy(), expected.to_numpy())


def test_transform_single_matches_transform_for_main_effects() -> None:
    """Single-row inference should produce the same main-effect values as
    the bulk transform on the same input."""
    df, _ = generate_synthetic_logs(SyntheticDataConfig(n_days=50, seed=0))
    fb = FeatureBuilder().fit(df)
    X = fb.transform(df)

    row = df.iloc[-1][BEHAVIOR_COLS].to_dict()
    single = fb.transform_single(row)
    feature_names = fb.feature_names_
    assert feature_names is not None
    for c in BEHAVIOR_COLS:
        assert np.isclose(single[feature_names.index(c)], X[c].iloc[-1])


def test_transform_single_with_history() -> None:
    """Lag values should be pulled from history, not zeroed out."""
    df, _ = generate_synthetic_logs(SyntheticDataConfig(n_days=50, seed=0))
    fb = FeatureBuilder(lags=[1, 2]).fit(df)
    X = fb.transform(df)

    # "Today" is the last row; "history" is the two preceding rows.
    today = df.iloc[-1][BEHAVIOR_COLS]
    history = df.iloc[-3:-1][BEHAVIOR_COLS]  # rows -3 and -2

    single = fb.transform_single_with_history(today, history)
    feature_names = fb.feature_names_
    assert feature_names is not None
    # sleep_hours_lag1 at this row should equal the standardized sleep from
    # yesterday (df.iloc[-2]).
    expected_lag1 = X["sleep_hours_lag1"].iloc[-1]
    assert np.isclose(single[feature_names.index("sleep_hours_lag1")], expected_lag1)
