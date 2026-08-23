"""Unit tests for field extractors — uses synthetic OCR tokens as input."""
from __future__ import annotations

from src.extractors.cost_extractor import CostExtractor, _parse_number
from src.extractors.dealer_extractor import DealerExtractor
from src.extractors.hp_extractor import HPExtractor
from src.extractors.model_extractor import ModelExtractor
from src.schemas.output import OCRToken
from src.utils.io import load_config


def _tok(text, x1, y1, x2, y2, conf=0.95):
    return OCRToken(text=text, bbox=[x1, y1, x2, y2], confidence=conf)


cfg = load_config("configs/default.yaml")


def test_dealer_extractor_fuzzy_match():
    tokens = [
        _tok("Dealer:", 50, 50, 130, 80),
        _tok("ABC Tractors Private Limited", 140, 50, 500, 80),
    ]
    res = DealerExtractor(cfg).extract(tokens)
    assert res.value is not None
    # Should match "ABC Tractors Pvt Ltd" in the master (fuzzy >= 90)
    assert "ABC Tractors" in res.value
    assert res.confidence >= 0.9


def test_model_extractor_exact_substring():
    tokens = [
        _tok("Model:", 50, 100, 130, 130),
        _tok("Mahindra 575 DI", 140, 100, 400, 130),
    ]
    res = ModelExtractor(cfg).extract(tokens)
    assert res.value == "Mahindra 575 DI"


def test_hp_extractor_regex():
    tokens = [
        _tok("Engine: 50 HP", 50, 200, 300, 230),
    ]
    res = HPExtractor(cfg).extract(tokens)
    assert res.value == 50


def test_hp_extractor_anchor_fallback():
    tokens = [
        _tok("HP", 50, 200, 80, 230),
        _tok("52", 100, 200, 140, 230),
    ]
    res = HPExtractor(cfg).extract(tokens)
    assert res.value == 52


def test_cost_extractor_grand_total_preferred():
    tokens = [
        _tok("Amount:", 50, 300, 150, 330),
        _tok("500000", 170, 300, 280, 330),
        _tok("Grand Total:", 50, 400, 200, 430),
        _tok("525000", 220, 400, 340, 430),
    ]
    res = CostExtractor(cfg).extract(tokens)
    assert res.value == 525000


def test_parse_number_currency_formats():
    assert _parse_number("Rs. 5,25,000") == 525000
    assert _parse_number("525000.00") == 525000
    assert _parse_number("₹ 1,00,000 /-") == 100000
    assert _parse_number("abc") is None


def test_cost_extractor_rejects_out_of_range():
    """A tiny number (like 12) shouldn't be mistaken for the cost."""
    tokens = [
        _tok("Total:", 50, 300, 120, 330),
        _tok("12", 140, 300, 170, 330),            # below min_cost
        _tok("525000", 200, 300, 300, 330),        # valid
    ]
    res = CostExtractor(cfg).extract(tokens)
    assert res.value == 525000
