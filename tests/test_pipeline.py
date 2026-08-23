"""
Pipeline smoke test — runs the pipeline against a tiny synthetic image and
asserts the output JSON has the expected schema.

OCR is the slowest dependency; we stub it for this test so it runs in CI
without requiring PaddleOCR to download ~300MB of model weights.
"""
from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.schemas.output import OCRToken


def _make_invoice_image():
    img = np.ones((1100, 800, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Dealer: ABC Tractors Pvt Ltd", (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Model: Mahindra 575 DI", (40, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "50 HP", (40, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Grand Total: Rs. 525000", (40, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


@pytest.fixture
def invoice_image(tmp_path):
    img = _make_invoice_image()
    p = tmp_path / "invoice.png"
    cv2.imwrite(str(p), img)
    return p


def _mock_ocr_tokens(image, page=0):
    return [
        OCRToken(text="Dealer:", bbox=[40, 80, 120, 110], confidence=0.97),
        OCRToken(text="ABC Tractors Private Limited",
                 bbox=[130, 80, 560, 110], confidence=0.95),
        OCRToken(text="Model:", bbox=[40, 180, 120, 210], confidence=0.97),
        OCRToken(text="Mahindra 575 DI", bbox=[130, 180, 400, 210], confidence=0.96),
        OCRToken(text="50 HP", bbox=[40, 280, 140, 310], confidence=0.98),
        OCRToken(text="Grand Total:", bbox=[40, 380, 210, 410], confidence=0.97),
        OCRToken(text="Rs. 525000", bbox=[220, 380, 430, 410], confidence=0.96),
    ]


def test_pipeline_end_to_end(invoice_image):
    from src.pipeline import InvoicePipeline

    pipe = InvoicePipeline()
    # Stub OCR
    with patch.object(
        pipe.__class__, "ocr",
        new=type("MockOCR", (), {"run": staticmethod(_mock_ocr_tokens)})(),
    ):
        result = pipe.run(str(invoice_image), doc_id="t1")

    d = result.model_dump()
    assert d["doc_id"] == "t1"
    assert "Mahindra 575 DI" in str(d["fields"]["model_name"])
    assert d["fields"]["horse_power"] == 50
    assert d["fields"]["asset_cost"] == 525000
    assert "dealer_name" in d["fields"]
    assert d["processing_time_sec"] >= 0
    assert d["cost_estimate_usd"] >= 0
    assert 0.0 <= d["confidence"] <= 1.0
