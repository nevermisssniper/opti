"""Optimizer tests. Each test corresponds to a property we claim about the
constrained QP."""
from __future__ import annotations

from opti.data import SyntheticDataConfig, generate_synthetic_logs
from opti.features import FeatureBuilder
from opti.model import OptiModel
from opti.optimizer import BehaviorOptimizer


def _build_full_pipeline(n_days: int = 200, seed: int = 0):
    cfg = SyntheticDataConfig(n_days=n_days, seed=seed)
    df, _ = generate_synthetic_logs(cfg)
    fb = FeatureBuilder().fit(df)
    X, y = fb.transform_with_target(df)
    model = OptiModel().fit(X, y)
    opt = BehaviorOptimizer(model=model, features=fb, bounds=cfg.bounds)
    return df, fb, model, opt, cfg


def test_recommendation_satisfies_targets() -> None:
    """If we ask for achievable targets, the post-optimization predictions
    should meet or exceed them."""
    df, fb, model, opt, cfg = _build_full_pipeline()
    today = df.iloc[-1][fb.behavior_cols].to_dict()
    history = df.iloc[-3:][fb.behavior_cols]
    targets = {"mental_score": 65.0, "physical_score": 65.0, "cognitive_score": 65.0}
    result = opt.recommend(today, targets, history=history)
    assert result.success
    for k, tgt in targets.items():
        # Allow a tiny tolerance for SLSQP's inequality slack.
        assert result.predicted_after[k] >= tgt - 1e-4, (
            f"{k}: predicted {result.predicted_after[k]:.3f} < target {tgt}"
        )


def test_recommendation_respects_bounds() -> None:
    df, fb, model, opt, cfg = _build_full_pipeline()
    today = df.iloc[-1][fb.behavior_cols].to_dict()
    targets = {"mental_score": 70.0, "physical_score": 70.0, "cognitive_score": 70.0}
    result = opt.recommend(today, targets, history=df.iloc[-3:][fb.behavior_cols])
    for col, (lo, hi) in cfg.bounds.items():
        v = result.recommended_behavior[col]
        assert lo - 1e-6 <= v <= hi + 1e-6, f"{col}={v:.3f} outside [{lo},{hi}]"


def test_no_change_when_targets_already_met() -> None:
    """If today's predicted performance already meets the targets, optimal
    delta should be ~zero (effort cost minimized)."""
    df, fb, model, opt, cfg = _build_full_pipeline()
    today = df.iloc[-1][fb.behavior_cols].to_dict()
    history = df.iloc[-3:][fb.behavior_cols]
    # First, predict today's performance.
    targets_floor = {"mental_score": 0.0, "physical_score": 0.0, "cognitive_score": 0.0}
    baseline = opt.recommend(today, targets_floor, history=history)
    assert baseline.effort_norm < 1e-6


def test_fixed_features_dont_change() -> None:
    """If we mark training_load as fixed (already done for the day), its
    delta must be exactly zero regardless of constraint pressure."""
    df, fb, model, opt_default, cfg = _build_full_pipeline()
    opt_fixed = BehaviorOptimizer(
        model=model, features=fb, bounds=cfg.bounds, fixed=["training_load"]
    )
    today = df.iloc[-1][fb.behavior_cols].to_dict()
    targets = {"mental_score": 75.0, "physical_score": 75.0, "cognitive_score": 75.0}
    result = opt_fixed.recommend(today, targets, history=df.iloc[-3:][fb.behavior_cols])
    assert abs(result.delta["training_load"]) < 1e-9


def test_higher_targets_require_more_effort() -> None:
    """Monotonicity: raising the performance floor must (weakly) increase
    the effort norm."""
    df, fb, model, opt, cfg = _build_full_pipeline()
    today = df.iloc[-1][fb.behavior_cols].to_dict()
    history = df.iloc[-3:][fb.behavior_cols]

    efforts = []
    for tgt in [50.0, 60.0, 65.0, 70.0]:
        result = opt.recommend(
            today,
            {"mental_score": tgt, "physical_score": tgt, "cognitive_score": tgt},
            history=history,
        )
        if result.success:
            efforts.append(result.effort_norm)
    # Should be (weakly) non-decreasing.
    for a, b in zip(efforts, efforts[1:]):
        assert b >= a - 1e-6, f"Effort decreased from {a:.4f} to {b:.4f}"
