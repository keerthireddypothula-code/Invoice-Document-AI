"""
Evaluation metrics.

Implements the exact metrics mentioned in the problem statement and CV:
    - Document-Level Accuracy (DLA) — % docs where all 6 fields correct
    - IoU for bounding boxes (signature / stamp) — CV claim: IoU > 0.5
    - Field-level mAP@0.5–0.95 for detections
    - Fuzzy match (≥90%) for dealer_name — CV claim
    - Exact / ±5%-tolerance for numeric fields
    - Per-field precision / recall / F1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz


# Abbreviation normalization for Indian business names — applied to both
# pred and GT before fuzzy comparison so "Pvt Ltd" ↔ "Private Limited"
# doesn't count as a mismatch.
_DEALER_ABBREV = {
    "pvt": "private",
    "ltd": "limited",
    "co.": "company",
    "co": "company",
    "&": "and",
    "m/s": "",
    "messrs": "",
}


def _canon_dealer(s: str) -> str:
    import re
    out = s.lower().strip()
    for pat, repl in _DEALER_ABBREV.items():
        out = re.sub(rf"\b{re.escape(pat)}\b", repl, out)
    out = re.sub(r"\s+", " ", out).strip(" .,:-")
    return out


# ───────── IoU ─────────

def iou(box_a: List[int], box_b: List[int]) -> float:
    """Compute IoU of two [x1, y1, x2, y2] bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


# ───────── Field-level match predicates ─────────

def text_exact_match(pred: Optional[str], gt: Optional[str]) -> bool:
    if pred is None or gt is None:
        return False
    return pred.strip().lower() == gt.strip().lower()


def text_fuzzy_match(
    pred: Optional[str], gt: Optional[str], threshold: int = 90
) -> bool:
    if pred is None or gt is None:
        return False
    # Canonicalize business abbreviations before scoring
    return fuzz.WRatio(_canon_dealer(pred), _canon_dealer(gt)) >= threshold


def numeric_match(
    pred: Optional[float], gt: Optional[float], tol_pct: float = 5.0
) -> bool:
    """Numeric match within ±tol_pct%."""
    if pred is None or gt is None:
        return False
    if gt == 0:
        return pred == 0
    return abs(pred - gt) / abs(gt) * 100.0 <= tol_pct


def presence_bbox_match(
    pred_present: bool,
    pred_bbox: Optional[List[int]],
    gt_present: bool,
    gt_bbox: Optional[List[int]],
    iou_thresh: float = 0.5,
) -> bool:
    """Presence flag must agree AND if both present, IoU >= threshold."""
    if pred_present != gt_present:
        return False
    if not gt_present:
        return True
    if pred_bbox is None or gt_bbox is None:
        return False
    return iou(pred_bbox, gt_bbox) >= iou_thresh


# ───────── Document-level metric ─────────

FIELD_SPECS = {
    "dealer_name": ("text_fuzzy", 90),
    "model_name":  ("text_exact", None),
    "horse_power": ("numeric", 5.0),
    "asset_cost":  ("numeric", 5.0),
    "signature":   ("presence_bbox", 0.5),
    "stamp":       ("presence_bbox", 0.5),
}


@dataclass
class FieldResult:
    field: str
    correct: bool
    pred_value: object = None
    gt_value: object = None


@dataclass
class DocumentResult:
    doc_id: str
    fields: List[FieldResult] = field(default_factory=list)

    @property
    def all_correct(self) -> bool:
        return all(f.correct for f in self.fields)

    @property
    def num_correct(self) -> int:
        return sum(1 for f in self.fields if f.correct)


def evaluate_document(pred: dict, gt: dict) -> DocumentResult:
    """Compare a predicted document to its ground truth."""
    out = DocumentResult(doc_id=pred.get("doc_id", gt.get("doc_id", "?")))
    p_fields = pred.get("fields", {})
    g_fields = gt.get("fields", {})

    for name, spec in FIELD_SPECS.items():
        kind, param = spec
        p_val = p_fields.get(name)
        g_val = g_fields.get(name)

        if kind == "text_exact":
            ok = text_exact_match(p_val, g_val)
        elif kind == "text_fuzzy":
            ok = text_fuzzy_match(p_val, g_val, threshold=param)
        elif kind == "numeric":
            ok = numeric_match(
                p_val if isinstance(p_val, (int, float)) else None,
                g_val if isinstance(g_val, (int, float)) else None,
                tol_pct=param,
            )
        elif kind == "presence_bbox":
            pp = bool(p_val.get("present", False)) if isinstance(p_val, dict) else False
            pb = p_val.get("bbox") if isinstance(p_val, dict) else None
            gp = bool(g_val.get("present", False)) if isinstance(g_val, dict) else False
            gb = g_val.get("bbox") if isinstance(g_val, dict) else None
            ok = presence_bbox_match(pp, pb, gp, gb, iou_thresh=param)
        else:
            ok = False

        out.fields.append(
            FieldResult(field=name, correct=ok, pred_value=p_val, gt_value=g_val)
        )
    return out


# ───────── Aggregation ─────────

def aggregate_metrics(results: List[DocumentResult]) -> Dict:
    n = len(results)
    if n == 0:
        return {"dla": 0.0, "per_field_accuracy": {}, "n": 0}

    dla = sum(1 for r in results if r.all_correct) / n
    per_field_acc = {name: 0 for name in FIELD_SPECS}
    for r in results:
        for f in r.fields:
            if f.correct:
                per_field_acc[f.field] += 1
    per_field_acc = {k: v / n for k, v in per_field_acc.items()}
    return {
        "dla": dla,
        "per_field_accuracy": per_field_acc,
        "n": n,
    }


# ───────── mAP@[0.5:0.95] for detection boxes ─────────

def average_precision_at_iou(
    pred_boxes: List[Tuple[List[int], float]],   # (bbox, score)
    gt_boxes: List[List[int]],
    iou_thresh: float,
) -> float:
    """AP at a single IoU threshold using 11-point interpolation."""
    if not gt_boxes:
        return 0.0 if pred_boxes else 1.0
    if not pred_boxes:
        return 0.0

    pred_sorted = sorted(pred_boxes, key=lambda x: -x[1])
    matched_gt = set()
    tps = []
    fps = []
    for bbox, _score in pred_sorted:
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            cur = iou(bbox, g)
            if cur > best_iou:
                best_iou = cur
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            matched_gt.add(best_j)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    # Cumulative precision/recall
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []
    for tp, fp in zip(tps, fps):
        tp_cum += tp
        fp_cum += fp
        precisions.append(tp_cum / max(1, tp_cum + fp_cum))
        recalls.append(tp_cum / len(gt_boxes))

    # 11-point interpolated AP
    ap = 0.0
    for r_target in [i / 10 for i in range(11)]:
        prec_at = [p for p, r in zip(precisions, recalls) if r >= r_target]
        ap += max(prec_at) / 11 if prec_at else 0.0
    return min(1.0, ap)   # clamp tiny floating-point overshoot


def map_50_95(
    pred_boxes: List[Tuple[List[int], float]],
    gt_boxes: List[List[int]],
) -> float:
    """mAP averaged over IoU thresholds 0.5:0.05:0.95."""
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    aps = [average_precision_at_iou(pred_boxes, gt_boxes, t) for t in thresholds]
    return sum(aps) / len(aps)
