"""
Evaluator — runs the pipeline over a dataset with ground-truth JSONs and
produces a full metrics report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from src.evaluation.metrics import (
    DocumentResult,
    aggregate_metrics,
    evaluate_document,
    map_50_95,
)
from src.utils.io import load_json, save_json
from src.utils.logging import log


class Evaluator:
    """Run predictions + GT through the metrics engine."""

    def __init__(self, predictions_dir: str, ground_truth_dir: str):
        self.pred_dir = Path(predictions_dir)
        self.gt_dir = Path(ground_truth_dir)

    def run(self, output_report: str = "outputs/eval_report.json") -> Dict:
        pred_files = sorted(self.pred_dir.glob("*.json"))
        if not pred_files:
            log.warning(f"No prediction files found in {self.pred_dir}")
            return {"dla": 0.0, "n": 0}

        doc_results: List[DocumentResult] = []
        sig_preds, sig_gts = [], []
        stamp_preds, stamp_gts = [], []

        for pf in tqdm(pred_files, desc="evaluating"):
            pred = load_json(pf)
            gt_path = self.gt_dir / pf.name
            if not gt_path.exists():
                log.warning(f"No GT for {pf.name}; skipping")
                continue
            gt = load_json(gt_path)

            doc_results.append(evaluate_document(pred, gt))

            # Collect boxes for mAP
            p_sig = pred.get("fields", {}).get("signature", {})
            g_sig = gt.get("fields", {}).get("signature", {})
            if p_sig.get("present") and p_sig.get("bbox"):
                sig_preds.append((p_sig["bbox"], p_sig.get("confidence", 0.5)))
            if g_sig.get("present") and g_sig.get("bbox"):
                sig_gts.append(g_sig["bbox"])

            p_st = pred.get("fields", {}).get("stamp", {})
            g_st = gt.get("fields", {}).get("stamp", {})
            if p_st.get("present") and p_st.get("bbox"):
                stamp_preds.append((p_st["bbox"], p_st.get("confidence", 0.5)))
            if g_st.get("present") and g_st.get("bbox"):
                stamp_gts.append(g_st["bbox"])

        agg = aggregate_metrics(doc_results)
        sig_map = map_50_95(sig_preds, sig_gts)
        stamp_map = map_50_95(stamp_preds, stamp_gts)

        report = {
            **agg,
            "mAP_50_95": {
                "signature": sig_map,
                "stamp": stamp_map,
                "mean": (sig_map + stamp_map) / 2,
            },
            "per_doc": [
                {
                    "doc_id": r.doc_id,
                    "all_correct": r.all_correct,
                    "num_correct": r.num_correct,
                    "fields": [
                        {"field": f.field, "correct": f.correct}
                        for f in r.fields
                    ],
                }
                for r in doc_results
            ],
        }
        save_json(report, output_report)
        log.info(f"Saved eval report → {output_report}")
        log.info(
            f"DLA: {agg['dla']:.3f} ({int(agg['dla']*agg['n'])}/{agg['n']}) | "
            f"Signature mAP: {sig_map:.3f} | Stamp mAP: {stamp_map:.3f}"
        )
        return report
