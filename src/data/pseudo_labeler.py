"""
Pseudo-labeler — addresses the "no ground truth" setting in the problem
statement.

Workflow:
    1. Run the pipeline on every document.
    2. Split outputs into HIGH-confidence and LOW-confidence buckets based on
       per-field confidence thresholds.
    3. HIGH-confidence outputs become pseudo-labels (treated as silver truth).
    4. LOW-confidence outputs are queued for human review in the Streamlit UI.
    5. Corrected labels are written back and combined with pseudo-labels to
       form the final training/validation set.

This implements two of the four strategies recommended in the problem
statement: Pseudo-Labeling + Manual Sampling (active-learning-style).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils.io import save_json
from src.utils.logging import log


HIGH_CONF_FIELD_THRESHOLDS = {
    "dealer_name": 0.90,
    "model_name":  0.90,
    "horse_power": 0.85,
    "asset_cost":  0.85,
    "signature":   0.50,
    "stamp":       0.50,
}


def bucketize(prediction: dict) -> str:
    """Return 'high' if all fields meet thresholds, else 'low'."""
    confs = prediction.get("field_confidences", {})
    for name, thresh in HIGH_CONF_FIELD_THRESHOLDS.items():
        if confs.get(name, 0.0) < thresh:
            return "low"
    # Also require that no value is None
    fields = prediction.get("fields", {})
    for name in ("dealer_name", "model_name", "horse_power", "asset_cost"):
        if fields.get(name) in (None, ""):
            return "low"
    return "high"


def run_pseudo_labeling(
    predictions_dir: str, out_root: str = "outputs/pseudo_labels"
) -> Tuple[int, int]:
    """Partition predictions into high/low confidence buckets."""
    pred_dir = Path(predictions_dir)
    out_root = Path(out_root)
    high_dir = out_root / "high_confidence"
    low_dir = out_root / "for_review"
    high_dir.mkdir(parents=True, exist_ok=True)
    low_dir.mkdir(parents=True, exist_ok=True)

    high_cnt = 0
    low_cnt = 0
    for pf in pred_dir.glob("*.json"):
        with open(pf) as f:
            pred = json.load(f)
        bucket = bucketize(pred)
        target = high_dir if bucket == "high" else low_dir
        save_json(pred, target / pf.name)
        if bucket == "high":
            high_cnt += 1
        else:
            low_cnt += 1

    log.info(
        f"Pseudo-labeling done. HIGH={high_cnt} (auto-accepted) | "
        f"LOW={low_cnt} (needs review)."
    )
    return high_cnt, low_cnt
