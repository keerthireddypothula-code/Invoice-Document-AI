"""
Asset cost extractor.

Strategy:
    1. Anchor on tokens like 'total', 'grand total', 'asset cost', 'on road',
       'ex-showroom', and their Hindi/Gujarati equivalents.
    2. Parse numeric neighbors, stripping Rs./INR/commas/spaces.
    3. Validate against plausible tractor price range.
    4. If multiple anchors exist, prefer 'grand total' > 'total' > others.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.extractors.base import BaseExtractor, ExtractionResult
from src.schemas.output import OCRToken


# Preference order for anchor specificity (higher = more preferred)
_ANCHOR_PRIORITY = {
    "grand total": 5,
    "asset cost": 4,
    "on road": 4,
    "total": 3,
    "amount": 2,
    "price": 1,
    "ex-showroom": 3,
    "कुल": 3,
    "ટોટલ": 3,
    "કુલ": 3,
}


def _parse_number(s: str) -> Optional[int]:
    """Parse an Indian-formatted currency number like 'Rs. 5,25,000' → 525000."""
    # Remove currency symbols, INR, Rs., etc.
    cleaned = re.sub(r"(?i)(rs\.?|inr|₹|rupees?)", "", s)
    cleaned = re.sub(r"[^\d.]", "", cleaned)
    if not cleaned:
        return None
    # Take only the integer part (drop decimals for costs)
    integer = cleaned.split(".")[0]
    if not integer:
        return None
    try:
        return int(integer)
    except ValueError:
        return None


class CostExtractor(BaseExtractor):
    field_name = "asset_cost"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        ext = cfg.get("extraction", {}).get("cost", {})
        self.anchors: List[str] = ext.get("anchors", ["total", "amount"])
        self.min_cost = ext.get("min_cost", 50_000)
        self.max_cost = ext.get("max_cost", 5_000_000)

    def extract(self, tokens: List[OCRToken]) -> ExtractionResult:
        if not tokens:
            return ExtractionResult(None, 0.0, [])

        best: Optional[Tuple[int, float, int, List[int]]] = None
        # (value, ocr_conf, anchor_priority, evidence_idxs)

        for anchor in self.anchors:
            priority = _ANCHOR_PRIORITY.get(anchor.lower(), 1)
            for ai, t in enumerate(tokens):
                if anchor.lower() not in t.text.lower():
                    continue
                for j, _dist in self._neighbors_right_or_below(tokens, ai, 5):
                    val = _parse_number(tokens[j].text)
                    if val is None or not (self.min_cost <= val <= self.max_cost):
                        continue
                    conf = tokens[j].confidence
                    cand = (val, conf, priority, [ai, j])
                    if best is None or (cand[2], cand[1]) > (best[2], best[1]):
                        best = cand

        # Fallback: scan all tokens for any numeric that fits the range.
        # This covers invoices where the "Total" label is missing.
        if best is None:
            numeric_candidates = []
            for i, t in enumerate(tokens):
                val = _parse_number(t.text)
                if val is not None and self.min_cost <= val <= self.max_cost:
                    numeric_candidates.append((val, t.confidence, i))
            if numeric_candidates:
                # Pick the largest — in invoices the grand total is usually max
                v, c, i = max(numeric_candidates, key=lambda x: x[0])
                return ExtractionResult(v, c * 0.7, [i])

        if best is None:
            return ExtractionResult(None, 0.0, [])

        val, conf, _prio, idxs = best
        return ExtractionResult(val, conf, idxs)
