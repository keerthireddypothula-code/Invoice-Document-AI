"""
Evaluation script — runs the Evaluator over predictions + ground truth.

Usage:
    python scripts/evaluate.py \
        --predictions outputs/predictions \
        --ground_truth data/synthetic/ground_truth \
        --report outputs/eval_report.json
"""
from __future__ import annotations

import argparse

from src.evaluation.evaluator import Evaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground_truth", required=True)
    ap.add_argument("--report", default="outputs/eval_report.json")
    args = ap.parse_args()

    ev = Evaluator(args.predictions, args.ground_truth)
    report = ev.run(args.report)

    print("\n=== Evaluation Summary ===")
    print(f"Documents evaluated:     {report['n']}")
    print(f"Document-Level Accuracy: {report['dla']:.3f}")
    print("\nPer-field accuracy:")
    for field, acc in report.get("per_field_accuracy", {}).items():
        print(f"  {field:<15}: {acc:.3f}")
    print("\nmAP@50-95:")
    for cls, val in report.get("mAP_50_95", {}).items():
        print(f"  {cls:<15}: {val:.3f}")


if __name__ == "__main__":
    main()
