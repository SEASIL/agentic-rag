"""
CI entry point. Runs Ragas evaluation and exits non-zero if any metric falls
below its threshold in configs/settings.py — this is what gates PRs in
.github/workflows/eval.yml.

Usage:
    python scripts/run_eval_ci.py                      # golden set, absolute thresholds
    python scripts/run_eval_ci.py --testset path.jsonl
    python scripts/run_eval_ci.py --baseline baseline_scores.json  # regression-only mode
"""
from __future__ import annotations

import argparse
import json
import sys

from configs.settings import settings
from src.eval.ragas_pipeline import run_evaluation

# Allowed regression margin when comparing against a baseline (regression-only mode)
REGRESSION_TOLERANCE = 0.02


def check_absolute_thresholds(scores: dict) -> list[str]:
    failures = []
    for metric, threshold in settings.ragas_thresholds.items():
        score = scores.get(metric)
        if score is None:
            failures.append(f"  {metric}: MISSING from results")
        elif score < threshold:
            failures.append(f"  {metric}: {score:.3f} < required {threshold:.3f}")
    return failures


def check_regression(scores: dict, baseline: dict) -> list[str]:
    failures = []
    for metric, score in scores.items():
        base = baseline.get(metric)
        if base is None:
            continue
        if score < base - REGRESSION_TOLERANCE:
            failures.append(f"  {metric}: {score:.3f} regressed from baseline {base:.3f}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="src/eval/golden_testset.jsonl")
    parser.add_argument("--baseline", default=None, help="Path to baseline_scores.json for regression-only mode")
    parser.add_argument("--out", default="eval_results.json")
    args = parser.parse_args()

    print(f"Running Ragas evaluation on {args.testset} ...")
    scores = run_evaluation(args.testset)

    with open(args.out, "w") as f:
        json.dump(scores, f, indent=2)

    print("\n=== Scores ===")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.3f}")

    if args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)
        failures = check_regression(scores, baseline)
        mode = "regression-only"
    else:
        failures = check_absolute_thresholds(scores)
        mode = "absolute threshold"

    if failures:
        print(f"\nFAILED ({mode} mode):")
        print("\n".join(failures))
        sys.exit(1)

    print(f"\nPASSED ({mode} mode).")


if __name__ == "__main__":
    main()
