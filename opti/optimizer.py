"""Constrained behavioral optimization.

Given a fitted linear model and today's behavior x_0, find Delta x such that:

    minimize    ||Delta x||_2^2                              (effort cost)
    subject to  W^T (x_0 + Delta x) + b >= target_perf       (performance floor)
                lb <= x_0 + Delta x <= ub                    (physiological bounds)
                Delta x[fixed] = 0                           (immutable features)

This is a constrained nonlinear optimization problem:
  - Objective is positive-definite quadratic in Delta x.
  - Interaction features make the performance constraints nonlinear.

We solve it with scipy's SLSQP. For a problem this small (~8-12 mutable
behavioral inputs) SLSQP is overkill but reliable; we'd swap to cvxpy or a
proper QP solver if scaling up.

Why minimize effort (||Delta x||^2) rather than maximize predicted score?
  Two reasons:
  1. The maximize-score formulation has degenerate solutions: push every
     behavioral knob to its bound in the direction of its coefficient.
     That's not actionable advice.
  2. Behavioral interventions have implicit cost. The user wants the
     SMALLEST change that hits a performance floor, not the change that
     maximizes a model that's only an approximation anyway. This also
     makes the recommendations robust to model misspecification: small
     Delta means we don't extrapolate far from training distribution.

Lagged features are treated as fixed: we can't change yesterday. They
contribute to the constraint as constants.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opti.data import BEHAVIOR_COLS
from opti.features import FeatureBuilder
from opti.model import OptiModel


@dataclass
class OptimizationResult:
    """Result of a single optimization call."""

    success: bool
    message: str
    current_behavior: dict[str, float]
    recommended_behavior: dict[str, float]
    delta: dict[str, float]
    predicted_before: dict[str, float]
    predicted_after: dict[str, float]
    effort_norm: float  # sqrt of sum of squared standardized deltas


class BehaviorOptimizer:
    """Solve the constrained nonlinear optimization that surfaces minimum behavioral changes.

    Parameters
    ----------
    model : OptiModel
        Fitted multi-output Ridge.
    features : FeatureBuilder
        Fitted feature builder. We need its means_/stds_ to convert between
        raw behavior units and the standardized space the model lives in.
    bounds : dict
        Physiological lower/upper bounds per behavioral input (raw units).
    fixed : list[str] | None
        Behavioral inputs the user cannot change (e.g., we might fix
        training_load because today's session is already done).
    """

    def __init__(
        self,
        model: OptiModel,
        features: FeatureBuilder,
        bounds: dict[str, tuple[float, float]],
        fixed: list[str] | None = None,
    ) -> None:
        if model.feature_names_ is None or model.target_names_ is None:
            raise RuntimeError("Model must be fit before optimization.")
        if features.means_ is None or features.stds_ is None:
            raise RuntimeError("FeatureBuilder must be fit before optimization.")
        self.model = model
        self.features = features
        self.bounds = bounds
        self.fixed = fixed or []
        self.behavior_cols = list(BEHAVIOR_COLS)

    def recommend(
        self,
        current_behavior: dict[str, float] | pd.Series,
        target_perf: dict[str, float],
        history: pd.DataFrame | None = None,
    ) -> OptimizationResult:
        """Find minimum-effort changes that meet target performance per domain.

        Parameters
        ----------
        current_behavior : mapping
            Today's behavior in raw units (sleep_hours=7.0, hrv_ms=55.0, ...).
        target_perf : mapping
            Minimum acceptable score per output (mental_score=70.0, ...).
            Outputs not specified are unconstrained.
        history : DataFrame | None
            Recent days of behavior (raw units) for lag features. If None,
            lag features default to the training-set mean (zero after
            standardization).
        """
        if isinstance(current_behavior, dict):
            current_behavior = pd.Series(current_behavior)

        # Build the feature vector once at x_0 to get the "constant" lag &
        # interaction structure. We'll perturb only the main-effect entries.
        if history is not None:
            x0_feat = self.features.transform_single_with_history(
                current_behavior, history
            )
        else:
            x0_feat = self.features.transform_single(current_behavior)

        # The decision variables are deltas to the 8 raw behaviors in their
        # STANDARDIZED form (so the L2 objective is unit-balanced). We map
        # back to raw at the end.
        n_behaviors = len(self.behavior_cols)
        means = self.features.means_
        stds = self.features.stds_
        assert means is not None and stds is not None

        # Index lookups: where in the feature vector each behavior's main
        # effect lives, and the indices of its interaction terms.
        assert self.features.feature_names_ is not None
        feature_names = self.features.feature_names_
        main_idx = {c: feature_names.index(c) for c in self.behavior_cols}

        # For each interaction (a, b), store (idx_in_feature_vec, a, b).
        interaction_terms: list[tuple[int, str, str]] = []
        for a, b in self.features.interactions:
            name = f"{a}_x_{b}"
            if name in feature_names:
                interaction_terms.append((feature_names.index(name), a, b))

        # Current standardized behavior values (for reconstructing interactions
        # under perturbation).
        z0 = ((current_behavior[self.behavior_cols] - means) / stds).to_numpy()

        # Fixed-mask: which deltas are forced to zero.
        fixed_mask = np.array([c in self.fixed for c in self.behavior_cols], dtype=bool)

        # Bounds on delta in standardized units. From the raw bounds and
        # current value: lb_raw <= x_0 + delta_raw <= ub_raw
        #   => (lb_raw - x_0)/std <= delta_z <= (ub_raw - x_0)/std
        lb_z = np.zeros(n_behaviors)
        ub_z = np.zeros(n_behaviors)
        for i, c in enumerate(self.behavior_cols):
            lo, hi = self.bounds[c]
            lb_z[i] = (lo - current_behavior[c]) / stds[c]
            ub_z[i] = (hi - current_behavior[c]) / stds[c]
            if fixed_mask[i]:
                lb_z[i] = 0.0
                ub_z[i] = 0.0

        W = self.model.coef_       # (n_targets, n_features)
        b = self.model.intercept_  # (n_targets,)
        target_names = self.model.target_names_
        assert target_names is not None

        # Predicted performance as a function of delta_z.
        def feature_vec(delta_z: np.ndarray) -> np.ndarray:
            x = x0_feat.copy()
            # Update main effects
            for i, c in enumerate(self.behavior_cols):
                x[main_idx[c]] = z0[i] + delta_z[i]
            # Update interactions
            for idx, a, b_name in interaction_terms:
                ia = self.behavior_cols.index(a)
                ib = self.behavior_cols.index(b_name)
                za = z0[ia] + delta_z[ia]
                zb = z0[ib] + delta_z[ib]
                x[idx] = za * zb
            return x

        def predict_perf(delta_z: np.ndarray) -> np.ndarray:
            x = feature_vec(delta_z)
            return W @ x + b

        # Build per-output constraints in scipy's format.
        constraints = []
        for j, name in enumerate(target_names):
            if name not in target_perf:
                continue
            tgt = float(target_perf[name])

            def make_con(j_=j, tgt_=tgt):
                return lambda dz: float(predict_perf(dz)[j_] - tgt_)

            constraints.append({"type": "ineq", "fun": make_con()})

        # Objective: ||delta_z||^2 with gradient 2*delta_z.
        def objective(delta_z: np.ndarray) -> float:
            return float(np.dot(delta_z, delta_z))

        def objective_grad(delta_z: np.ndarray) -> np.ndarray:
            return 2.0 * delta_z

        x_init = np.zeros(n_behaviors)
        bounds_pairs = list(zip(lb_z.tolist(), ub_z.tolist(), strict=True))

        res = minimize(
            objective,
            x_init,
            jac=objective_grad,
            method="SLSQP",
            bounds=bounds_pairs,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-8},
        )

        delta_z_opt = res.x
        # Map back to raw behavioral units.
        delta_raw = delta_z_opt * stds.to_numpy()
        recommended_raw = current_behavior[self.behavior_cols].to_numpy() + delta_raw

        pred_before = W @ x0_feat + b
        pred_after = predict_perf(delta_z_opt)

        return OptimizationResult(
            success=bool(res.success),
            message=str(res.message),
            current_behavior={c: float(current_behavior[c]) for c in self.behavior_cols},
            recommended_behavior={
                c: float(v) for c, v in zip(self.behavior_cols, recommended_raw, strict=True)
            },
            delta={
                c: float(v) for c, v in zip(self.behavior_cols, delta_raw, strict=True)
            },
            predicted_before={
                name: float(pred_before[i]) for i, name in enumerate(target_names)
            },
            predicted_after={
                name: float(pred_after[i]) for i, name in enumerate(target_names)
            },
            effort_norm=float(np.sqrt(np.dot(delta_z_opt, delta_z_opt))),
        )

