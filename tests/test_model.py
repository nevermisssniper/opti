"""Model tests. The recovery test is the most important file in the project:
it proves the full pipeline (features -> model) recovers known ground-truth
coefficients within tolerance."""
from __future__ import annotations

import numpy as np

from opti.data import SyntheticDataConfig, generate_synthetic_logs
from opti.features import FeatureBuilder
from opti.model import OptiModel, _ridge_closed_form


def test_closed_form_matches_sklearn_via_fit() -> None:
    """OptiModel.fit() internally asserts closed-form == sklearn. If we get
    here without an exception, parity holds. This test exists to make that
    invariant visible at the test-suite level."""
    df, _ = generate_synthetic_logs(SyntheticDataConfig(n_days=150, seed=0))
    fb = FeatureBuilder().fit(df)
    X, y = fb.transform_with_target(df)
    OptiModel(alphas=(1.0,)).fit(X, y)  # no exception -> parity confirmed


def test_closed_form_unit_diagonal() -> None:
    """Sanity check on the closed-form solver: when X is the identity and y=X,
    Ridge returns I/(1+alpha)."""
    n = 5
    X = np.eye(n)
    y = X.copy()
    alpha = 2.0
    W = _ridge_closed_form(X, y, alpha)
    np.testing.assert_allclose(W, np.eye(n) / (1 + alpha))


def test_recovery_of_main_effects() -> None:
    """The model should recover the ground-truth main-effect coefficients
    within a tolerance set by noise + Ridge shrinkage.

    Ridge shrinks toward zero, so we expect recovered |coef| <= true |coef|.
    We check sign and rough magnitude.
    """
    df, truth = generate_synthetic_logs(SyntheticDataConfig(n_days=500, seed=0, noise_std=2.0))
    fb = FeatureBuilder().fit(df)
    X, y = fb.transform_with_target(df)
    model = OptiModel(alphas=(0.1, 1.0, 10.0)).fit(X, y)
    coef = model.coef_df()

    for target, expected_main in truth["main_effects"].items():
        for feature, true_w in expected_main.items():
            est = coef.loc[target, feature]
            assert np.sign(est) == np.sign(true_w), (
                f"{target} <- {feature}: estimate sign {est:.3f} disagrees with truth {true_w}"
            )
            # Magnitude within 50% (loose due to Ridge shrinkage + finite n).
            ratio = abs(est) / abs(true_w)
            assert 0.4 < ratio < 1.4, (
                f"{target} <- {feature}: |estimate|={abs(est):.3f} vs |truth|={abs(true_w)} "
                f"(ratio {ratio:.2f})"
            )


def test_recovery_of_interaction() -> None:
    """The sleep*hrv interaction is a key claim on the resume -- prove the
    model recovers it."""
    df, truth = generate_synthetic_logs(SyntheticDataConfig(n_days=500, seed=0, noise_std=2.0))
    fb = FeatureBuilder().fit(df)
    X, y = fb.transform_with_target(df)
    model = OptiModel(alphas=(0.1, 1.0, 10.0)).fit(X, y)
    coef = model.coef_df()

    for inter_name, per_target in truth["interactions"].items():
        for target, true_w in per_target.items():
            est = coef.loc[target, inter_name]
            assert np.sign(est) == np.sign(true_w)
            assert 0.3 < abs(est) / abs(true_w) < 1.5


def test_recovery_of_lagged_effects() -> None:
    """Lag-1 training_load -> physical_score and lag-2 review -> cognitive."""
    df, truth = generate_synthetic_logs(SyntheticDataConfig(n_days=500, seed=0, noise_std=2.0))
    fb = FeatureBuilder().fit(df)
    X, y = fb.transform_with_target(df)
    model = OptiModel(alphas=(0.1, 1.0, 10.0)).fit(X, y)
    coef = model.coef_df()

    for lag_name, per_target in truth["lags"].items():
        for target, true_w in per_target.items():
            est = coef.loc[target, lag_name]
            assert np.sign(est) == np.sign(true_w)


def test_chosen_alpha_is_from_grid() -> None:
    df, _ = generate_synthetic_logs(SyntheticDataConfig(n_days=200, seed=0))
    fb = FeatureBuilder().fit(df)
    X, y = fb.transform_with_target(df)
    alphas = (0.01, 0.1, 1.0, 10.0)
    model = OptiModel(alphas=alphas).fit(X, y)
    assert model.alpha_ in alphas


def test_r2_is_reasonable() -> None:
    """With known structure and 200 days of low-noise data, in-sample R^2
    should be solidly above 0.5 for all three outputs."""
    df, _ = generate_synthetic_logs(SyntheticDataConfig(n_days=200, seed=0, noise_std=2.0))
    fb = FeatureBuilder().fit(df)
    X, y = fb.transform_with_target(df)
    model = OptiModel().fit(X, y)
    scores = model.score(X, y)
    for target, r2 in scores.items():
        assert r2 > 0.5, f"{target} R^2 = {r2:.3f} too low"
