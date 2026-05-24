"""Smoke tests for synthetic data generation."""
from __future__ import annotations

import numpy as np

from opti.data import BEHAVIOR_COLS, PERFORMANCE_COLS, SyntheticDataConfig, generate_synthetic_logs


def test_shape_and_columns() -> None:
    df, _ = generate_synthetic_logs(SyntheticDataConfig(n_days=100, seed=0))
    assert len(df) == 100
    for c in BEHAVIOR_COLS + PERFORMANCE_COLS:
        assert c in df.columns


def test_behaviors_within_bounds() -> None:
    cfg = SyntheticDataConfig(n_days=200, seed=1)
    df, _ = generate_synthetic_logs(cfg)
    for col, (lo, hi) in cfg.bounds.items():
        assert df[col].min() >= lo - 1e-9, f"{col} below bound"
        assert df[col].max() <= hi + 1e-9, f"{col} above bound"


def test_performance_within_0_100() -> None:
    df, _ = generate_synthetic_logs(SyntheticDataConfig(n_days=200, seed=2))
    for c in PERFORMANCE_COLS:
        assert df[c].min() >= 0
        assert df[c].max() <= 100


def test_reproducibility() -> None:
    df1, _ = generate_synthetic_logs(SyntheticDataConfig(seed=7))
    df2, _ = generate_synthetic_logs(SyntheticDataConfig(seed=7))
    assert np.allclose(df1.to_numpy(), df2.to_numpy())
