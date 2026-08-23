"""Unit tests for evaluation metrics."""
from __future__ import annotations

from src.evaluation.metrics import (
    average_precision_at_iou,
    evaluate_document,
    iou,
    map_50_95,
    numeric_match,
    presence_bbox_match,
    text_exact_match,
    text_fuzzy_match,
)


def test_iou_identical_boxes():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_disjoint_boxes():
    assert iou([0, 0, 5, 5], [10, 10, 20, 20]) == 0.0


def test_iou_partial_overlap():
    # Two 10x10 boxes overlapping by 5x5 — IoU = 25 / (100 + 100 - 25) = 25/175
    assert abs(iou([0, 0, 10, 10], [5, 5, 15, 15]) - 25 / 175) < 1e-6


def test_iou_above_threshold():
    # CV claim validation: IoU > 0.5 works
    assert iou([0, 0, 100, 100], [10, 10, 100, 100]) > 0.5


def test_text_exact_match():
    assert text_exact_match("Mahindra 575 DI", "mahindra 575 di")
    assert not text_exact_match("Mahindra 575 DI", "Mahindra 415 DI")


def test_text_fuzzy_match_at_90():
    # CV claim: dealer fuzzy match ≥ 90%
    # WRatio scores minor OCR-style typos well.
    assert text_fuzzy_match("ABC Tractors Pvt Ltd", "ABC Tractor Pvt Ltd")
    assert text_fuzzy_match("Mahindra 575 DI", "Mahindra 575DI")
    # But clearly different names should NOT match.
    assert not text_fuzzy_match("ABC Tractors Pvt Ltd", "XYZ Motors")


def test_numeric_match_tolerance():
    assert numeric_match(525000, 525000)                # exact
    assert numeric_match(525000, 530000, tol_pct=5.0)   # within 5%
    assert not numeric_match(525000, 600000, tol_pct=5.0)


def test_presence_bbox_match():
    assert presence_bbox_match(True, [0, 0, 10, 10], True, [0, 0, 10, 10])
    assert not presence_bbox_match(True, [0, 0, 10, 10], True, [100, 100, 200, 200])
    assert presence_bbox_match(False, None, False, None)
    assert not presence_bbox_match(True, [0, 0, 10, 10], False, None)


def test_evaluate_document_all_correct():
    pred = {
        "doc_id": "d1",
        "fields": {
            "dealer_name": "ABC Tractors Pvt Ltd",
            "model_name": "Mahindra 575 DI",
            "horse_power": 50,
            "asset_cost": 525000,
            "signature": {"present": True, "bbox": [0, 0, 100, 50]},
            "stamp": {"present": True, "bbox": [200, 200, 300, 300]},
        },
    }
    gt = {
        "doc_id": "d1",
        "fields": {
            "dealer_name": "ABC Tractors Private Limited",
            "model_name": "Mahindra 575 DI",
            "horse_power": 50,
            "asset_cost": 525000,
            "signature": {"present": True, "bbox": [5, 5, 95, 45]},
            "stamp": {"present": True, "bbox": [205, 205, 295, 295]},
        },
    }
    res = evaluate_document(pred, gt)
    assert res.all_correct, f"Failed fields: {[f.field for f in res.fields if not f.correct]}"


def test_map_computation_runs():
    preds = [([0, 0, 10, 10], 0.9), ([100, 100, 120, 120], 0.8)]
    gts = [[0, 0, 10, 10]]
    ap = average_precision_at_iou(preds, gts, 0.5)
    assert 0.0 <= ap <= 1.0
    m = map_50_95(preds, gts)
    assert 0.0 <= m <= 1.0
