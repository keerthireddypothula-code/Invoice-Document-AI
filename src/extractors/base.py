"""
BaseExtractor — common utilities for all field extractors.

Each extractor receives the full list of OCR tokens and returns a
(value, confidence) tuple. Utilities here help locate anchor keywords
and pick nearby tokens (spatial association).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from src.schemas.output import OCRToken


@dataclass
class ExtractionResult:
    value: Optional[Any]
    confidence: float
    evidence_token_indices: List[int]


class BaseExtractor(ABC):
    """Abstract base class for a field extractor."""

    field_name: str = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def extract(self, tokens: List[OCRToken]) -> ExtractionResult:
        ...

    # ---- Helpers shared across concrete extractors ----

    @staticmethod
    def _find_anchor_indices(
        tokens: List[OCRToken], anchors: List[str]
    ) -> List[int]:
        """Return indices of tokens whose lowercase text contains any anchor."""
        out = []
        for i, t in enumerate(tokens):
            lower = t.text.lower()
            if any(a.lower() in lower for a in anchors):
                out.append(i)
        return out

    @staticmethod
    def _neighbors_right_or_below(
        tokens: List[OCRToken], anchor_idx: int, max_candidates: int = 5
    ) -> List[Tuple[int, float]]:
        """Return (idx, distance) of tokens to the right of or below an anchor.

        The 'right of or below' rule captures the typical key-value layout in
        Indian invoices ("Dealer: Foo" or "Dealer\\nFoo").
        """
        anchor = tokens[anchor_idx]
        ax1, ay1, ax2, ay2 = anchor.bbox
        anchor_cx = (ax1 + ax2) / 2
        anchor_cy = (ay1 + ay2) / 2

        scored = []
        for j, t in enumerate(tokens):
            if j == anchor_idx:
                continue
            tx1, ty1, tx2, ty2 = t.bbox
            tcx = (tx1 + tx2) / 2
            tcy = (ty1 + ty2) / 2

            # Right-of on same row
            same_row = abs(tcy - anchor_cy) < (ay2 - ay1) * 1.5
            right_of = tx1 > ax1

            # Below in roughly same column
            below = ty1 > ay2
            same_col = abs(tcx - anchor_cx) < (ax2 - ax1) * 2.5

            if (same_row and right_of) or (below and same_col):
                dist = ((tcx - anchor_cx) ** 2 + (tcy - anchor_cy) ** 2) ** 0.5
                scored.append((j, dist))
        scored.sort(key=lambda x: x[1])
        return scored[:max_candidates]
