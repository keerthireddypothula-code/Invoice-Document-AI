"""
Horsepower extractor.

Uses a combined anchor + regex strategy:
    - Primary: regex matches like '50 HP', '50HP', '50 bhp', '50 h.p.'
    - Secondary: anchor-token 'HP' with a numeric neighbor.
    - Tertiary: OCR-confusion fallback (S→5, O→0, I/l→1, B→8, Z→2)
      applied when the neighbor is alphabetic but in a plausible spot.

Validates that the extracted number is in a plausible tractor HP range
(defaults: 10..150). If multiple candidates found, prefers the one nearest
to an 'hp' anchor; otherwise the most confident OCR token.
"""
from __future__ import annotations

import re
from typing import List, Optional

from src.extractors.base import BaseExtractor, ExtractionResult
from src.schemas.output import OCRToken


# Common Tesseract / low-res OCR confusions for digits
_OCR_CHAR_FIX = str.maketrans({
    "S": "5", "s": "5",
    "O": "0", "o": "0",
    "I": "1", "l": "1",
    "B": "8",
    "Z": "2",
    "G": "6",
    "T": "7",
})


def _ocr_confused_to_int(s: str) -> Optional[int]:
    """Apply digit-confusion substitutions and try to parse an int."""
    fixed = s.translate(_OCR_CHAR_FIX)
    m = re.search(r"\d{1,3}", fixed)
    return int(m.group()) if m else None


class HPExtractor(BaseExtractor):
    field_name = "horse_power"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        ext = cfg.get("extraction", {}).get("horse_power", {})
        self.anchors = ext.get("anchors", ["hp", "horse power"])
        self.pattern = re.compile(
            ext.get("regex", r"(\d{1,3})\s*(?:hp|h\.p\.|bhp|horsepower)"),
            re.IGNORECASE,
        )
        # Second-pass regex using OCR-corrected string
        self.fuzzy_pattern = re.compile(
            r"([\dSsOoIlBZGT]{1,3})\s*(?:hp|h\.p\.|bhp|horsepower)",
            re.IGNORECASE,
        )
        self.min_hp = ext.get("min_hp", 10)
        self.max_hp = ext.get("max_hp", 150)

    def extract(self, tokens: List[OCRToken]) -> ExtractionResult:
        if not tokens:
            return ExtractionResult(None, 0.0, [])

        # 1) Regex pass over concatenated text
        joined = " ".join(t.text for t in tokens)
        matches = self.pattern.findall(joined)
        valid = [int(m) for m in matches if self.min_hp <= int(m) <= self.max_hp]
        if valid:
            return ExtractionResult(valid[0], 0.95, [])

        # 2) Anchor + numeric-neighbor (strict digits)
        anchor_idxs = self._find_anchor_indices(tokens, self.anchors)
        best_val: Optional[int] = None
        best_conf = 0.0
        best_idxs: List[int] = []
        for ai in anchor_idxs:
            for j, _ in self._neighbors_right_or_below(tokens, ai, 4):
                num = _first_int(tokens[j].text)
                if num is not None and self.min_hp <= num <= self.max_hp:
                    conf = tokens[j].confidence * 0.9
                    if conf > best_conf:
                        best_val = num
                        best_conf = conf
                        best_idxs = [ai, j]
        if best_val is not None:
            return ExtractionResult(best_val, best_conf, best_idxs)

        # 3) OCR-confusion fallback — second regex pass with substitutions
        fuzzy_matches = self.fuzzy_pattern.findall(joined)
        for m in fuzzy_matches:
            num = _ocr_confused_to_int(m)
            if num is not None and self.min_hp <= num <= self.max_hp:
                return ExtractionResult(num, 0.70, [])

        # 4) Anchor + OCR-confused neighbor
        for ai in anchor_idxs:
            for j, _ in self._neighbors_right_or_below(tokens, ai, 4):
                num = _ocr_confused_to_int(tokens[j].text)
                if num is not None and self.min_hp <= num <= self.max_hp:
                    return ExtractionResult(num, tokens[j].confidence * 0.65, [ai, j])

        return ExtractionResult(None, 0.0, [])


def _first_int(s: str) -> Optional[int]:
    m = re.search(r"\d{1,3}", s)
    return int(m.group()) if m else None

