# OPTI — Multi-Domain Performance Optimizer

A linear regression system that models cross-domain (mental / physical / cognitive) performance interactions, then surfaces the **minimum behavioral changes** required to hit a target performance floor via constrained gradient optimization.

```
behavior log ──▶ features (interactions + lags) ──▶ multi-output Ridge ──▶ constrained nonlinear optimization ──▶ recommendation
```

## What's in the box

| Component | File | Role |
|---|---|---|
| Synthetic data | `opti/data.py` | Daily logs with **known ground-truth weights** (the recovery test relies on this) |
| Features | `opti/features.py` | z-score → hand-picked interactions → t-1/t-2/t-3 lags |
| Model | `opti/model.py` | Multi-output Ridge with time-series-CV alpha selection; numpy closed-form parity check |
| Optimizer | `opti/optimizer.py` | SLSQP-solved constrained nonlinear optimization: minimize ‖Δx‖² s.t. predicted score ≥ target |
| CLI | `opti/cli.py` | `opti demo` runs the full pipeline end-to-end |
| Tests | `tests/` | 22 tests including coefficient-recovery on synthetic ground truth |

## Quick start

```bash
pip install -e ".[dev]"
pytest                  # 22 passing
python -m opti.cli demo # end-to-end recommendation
```

Sample output:
```
OPTI -- chosen Ridge alpha = 1
In-sample R^2: mental_score=0.952, physical_score=0.956, cognitive_score=0.917

Recommended behavior:
  ↑ sleep_hours              9.01  (Δ +0.60)
  ↑ hrv_ms                  53.07  (Δ +4.85)
  ↑ study_minutes           59.38  (Δ +21.48)
  ↓ screen_time            164.12  (Δ -28.23)
  ...

Predicted performance:
  mental_score      62.51  ->   72.00  (Δ +9.49)
  physical_score    59.30  ->   70.00  (Δ +10.70)
  cognitive_score   59.46  ->   68.00  (Δ +8.54)

Effort norm (||Δz||_2): 1.143
```

## Design decisions, defended

### Why linear?
- **Interpretability**: every coefficient corresponds to a single behavior or interaction the user can act on.
- **Sample efficiency**: realistic daily-log datasets are O(100) rows. Linear has the right inductive bias for that scale.
- **Differentiability**: the optimizer needs gradients of the prediction w.r.t. the input. A linear model gives them in closed form (the weight matrix is the Jacobian); a neural net would mean autograd through the network at every QP iteration.

### Why Ridge over OLS?
The design matrix has main effects, products of main effects, and lagged versions of all of them — ~40 features that are heavily collinear. OLS variance is unstable in that regime. Ridge shrinks toward zero in low-signal directions, trading a controlled bias for a large variance reduction. Alpha is picked by `TimeSeriesSplit` CV over `[0.01, 0.1, 1, 10, 100]`.

### Why time-series CV?
Rows have temporal dependence through lagged features: row t contains lag1 of row t−1. Random K-fold would split row t−1 into validation while leaving row t (which contains a lagged copy of t−1's behaviors) in training — target leakage. `TimeSeriesSplit` keeps validation strictly in the future of training.

### Why minimize ‖Δx‖ rather than maximize score?
Two reasons. First, the maximize-score formulation is degenerate: push every knob to its bound in the direction of its coefficient. That's not actionable. Second, behavior change has implicit cost — the user wants the *smallest* change that hits a floor, not the change that maxes out an approximate model. Small Δ also keeps the recommendation near the training distribution, where the linear model is more trustworthy.

### Why hand-picked interactions, not exhaustive polynomial expansion?
8 main features × full pairwise expansion = 28 interactions, most of which are noise at our sample size. We keep 5 that have a behavioral story (sleep×HRV = recovery, screen×sleep = sleep hygiene, etc.). Each one we keep is one we can defend in an interview.

### Why fixed-feature support in the optimizer?
At the time you ask for a recommendation today, some inputs are already determined: yesterday's training load, today's accumulated screen time, etc. The optimizer takes a `fixed` list and pins those deltas to zero so it can't recommend the impossible.

## The optimization problem

For a fitted linear model with weights `W ∈ ℝ^{T×F}` and intercept `b ∈ ℝ^T`:

```
minimize_{Δz}    ‖Δz‖₂²
subject to       (W φ(z₀ + Δz))_j + b_j  ≥  target_j         ∀ j ∈ constrained targets
                 lb_j ≤ (z₀)_j + (Δz)_j ≤ ub_j               ∀ j (mapped from raw bounds)
                 (Δz)_j = 0                                  ∀ j ∈ fixed
```

where `φ` re-derives interactions from the perturbed main effects so that `sleep × HRV` updates consistently with `sleep` and `HRV`. Lagged features are constants — yesterday is yesterday. The objective is quadratic, but recomputed interaction terms make the constraints nonlinear and potentially nonconvex. SLSQP returns a local solution; it does not guarantee a global minimum.

## Coefficient recovery (the headline test)

`tests/test_model.py::test_recovery_of_*` generate synthetic data with **known** weights, run the full feature → model pipeline, and assert that the recovered Ridge coefficients match the ground truth in sign and magnitude (within 50% — wider than the noise floor would imply because Ridge intentionally shrinks). This is the test that proves the resume bullets are real, not aspirational.

```bash
pytest tests/test_model.py -v
```

## What I would do next

In rough priority order:

1. **Replace synthetic data with real logs.** Same pipeline; nothing else changes.
2. **Bayesian Ridge / Gaussian Process** for predictive uncertainty intervals. Currently the optimizer treats `W @ x + b ≥ target` as a hard constraint with no uncertainty — a `≥ target` at 50% posterior probability is very different from one at 90%.
3. **Per-user hierarchical model.** Multi-task pooling across users would help with the small-sample-per-user problem.
4. **Reduced-rank multi-output regression.** Mental / physical / cognitive scores are correlated; we currently fit independent column-wise regressions, so we leave information on the table.
5. **Cross-output regularization** in the optimizer (Pareto frontier instead of fixed targets).

## Repo layout

```
opti/
├── opti/
│   ├── __init__.py
│   ├── data.py          # synthetic logs + ground-truth weights
│   ├── features.py      # FeatureBuilder (fit / transform / transform_single)
│   ├── model.py         # OptiModel (Ridge + closed-form parity)
│   ├── optimizer.py     # BehaviorOptimizer (SLSQP)
│   └── cli.py           # `opti demo`
├── tests/               # 22 tests, all passing
├── pyproject.toml       # package + ruff + mypy + pytest config
└── Makefile             # make test / make lint / make demo
```



## Interactive workbench (2026-11-13)

Run `python serve_demo.py`, then open <http://127.0.0.1:8000>. See [QUICKSTART.md](QUICKSTART.md) for core commands, tests, provenance and remaining requirements.
