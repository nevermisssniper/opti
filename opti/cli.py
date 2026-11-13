"""Minimal CLI: `opti demo` runs the full pipeline end-to-end and prints a
recommendation. Useful for showing in an interview without having to set up
a notebook."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from opti.data import SyntheticDataConfig, generate_synthetic_logs
from opti.features import FeatureBuilder
from opti.model import OptiModel
from opti.optimizer import BehaviorOptimizer


def _pretty_print(result: Any, model_scores: dict[str, float], alpha: float) -> None:
    print(f"\n{'='*60}")
    print(f"  OPTI -- chosen Ridge alpha = {alpha:g}")
    print("  In-sample R^2: " + ", ".join(f"{k}={v:.3f}" for k, v in model_scores.items()))
    print(f"{'='*60}\n")

    print("Current behavior:")
    for k, v in result.current_behavior.items():
        print(f"  {k:22s} {v:>8.2f}")

    print("\nRecommended behavior:")
    for k, v in result.recommended_behavior.items():
        delta = result.delta[k]
        marker = "  " if abs(delta) < 1e-6 else ("↑ " if delta > 0 else "↓ ")
        print(f"  {marker}{k:20s} {v:>8.2f}  (Δ {delta:+.2f})")

    print("\nPredicted performance:")
    for k in result.predicted_before:
        before = result.predicted_before[k]
        after = result.predicted_after[k]
        print(f"  {k:22s} {before:>6.2f}  ->  {after:>6.2f}  (Δ {after - before:+.2f})")

    print(f"\nEffort norm (||Δz||_2): {result.effort_norm:.4f}")
    print(f"Solver: {result.message}\n")


def run_demo(
    n_days: int,
    seed: int,
    targets: dict[str, float],
    output_json: bool,
) -> int:
    cfg = SyntheticDataConfig(n_days=n_days, seed=seed)
    df, _truth = generate_synthetic_logs(cfg)

    fb = FeatureBuilder().fit(df)
    X, y = fb.transform_with_target(df)

    model = OptiModel().fit(X, y)
    scores = model.score(X, y)

    # Use the most recent row as "today's behavior" and the prior days as history.
    history = df.iloc[:-1].tail(max(fb.lags))[fb.behavior_cols] if fb.lags else None
    today = df.iloc[-1][fb.behavior_cols].to_dict()

    opt = BehaviorOptimizer(model=model, features=fb, bounds=cfg.bounds)
    result = opt.recommend(today, target_perf=targets, history=history)

    if output_json:
        json.dump(
            {
                "alpha": model.alpha_,
                "r2": scores,
                "result": {
                    "success": result.success,
                    "current": result.current_behavior,
                    "recommended": result.recommended_behavior,
                    "delta": result.delta,
                    "predicted_before": result.predicted_before,
                    "predicted_after": result.predicted_after,
                    "effort_norm": result.effort_norm,
                },
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        _pretty_print(result, scores, model.alpha_ or 0.0)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opti", description="Multi-Domain Performance Optimizer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="Run a synthetic end-to-end demo")
    demo.add_argument("--days", type=int, default=180)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--mental", type=float, default=72.0)
    demo.add_argument("--physical", type=float, default=70.0)
    demo.add_argument("--cognitive", type=float, default=68.0)
    demo.add_argument("--json", action="store_true", help="Output JSON instead of pretty text")

    args = parser.parse_args(argv)

    if args.cmd == "demo":
        return run_demo(
            n_days=args.days,
            seed=args.seed,
            targets={
                "mental_score": args.mental,
                "physical_score": args.physical,
                "cognitive_score": args.cognitive,
            },
            output_json=args.json,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())

