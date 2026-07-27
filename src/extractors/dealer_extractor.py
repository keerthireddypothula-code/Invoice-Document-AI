"""
Dealer name extractor.

Strategy:
    1. Find anchor tokens matching {"dealer", "m/s", ...}.
    2. Build candidate phrases from adjacent tokens (right-of / below anchor).
    3. Fuzzy-match each candidate against the dealer column in asset_master.csv.
    4. Return the best match if score >= fuzzy_threshold (CV claim: ≥90%).

If no anchor is found, fall back to fuzzy-matching every contiguous trigram
of the document against the master list. Slower but robust for layouts
that don't have explicit "Dealer:" labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from rapidfuzz import fuzz, process

from src.extractors.base import BaseExtractor, ExtractionResult
from src.schemas.output import OCRToken


# Common abbreviation equivalents in Indian business names.
# Applied before fuzzy comparison so "Pvt Ltd" ↔ "Private Limited" scores high.
_ABBREV_MAP = {
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bco\b": "company",
    r"\bco\.\b": "company",
    r"\&": "and",
    r"m/s": "",
    r"messrs": "",
}


def _canonicalize(s: str) -> str:
    import re as _re
    out = s.lower().strip()
    for pat, repl in _ABBREV_MAP.items():
        out = _re.sub(pat, repl, out)
    out = _re.sub(r"\s+", " ", out).strip(" .,:-")
    return out


class DealerExtractor(BaseExtractor):
    field_name = "dealer_name"

    def __init__(self, cfg: dict, asset_master_path: Optional[str] = None):
        super().__init__(cfg)
        ext_cfg = cfg.get("extraction", {}).get("dealer", {})
        self.anchors: List[str] = ext_cfg.get("anchors", ["dealer"])
        self.threshold: int = ext_cfg.get("fuzzy_threshold", 90)

        master_path = asset_master_path or cfg.get("paths", {}).get(
            "asset_master", "configs/asset_master.csv"
        )
        self.dealer_list: List[str] = []
        self._canon_to_original: dict[str, str] = {}
        if Path(master_path).exists():
            df = pd.read_csv(master_path)
            for d in df["dealer_name"].dropna().unique().tolist():
                self.dealer_list.append(d)
                self._canon_to_original[_canonicalize(d)] = d

    def extract(self, tokens: List[OCRToken]) -> ExtractionResult:
        if not tokens:
            return ExtractionResult(None, 0.0, [])

        # 1) Anchor-driven candidates
        anchor_idxs = self._find_anchor_indices(tokens, self.anchors)
        candidates: List[tuple[str, List[int]]] = []
        for ai in anchor_idxs:
            neighbors = self._neighbors_right_or_below(tokens, ai, max_candidates=6)
            # Build phrases of length 1..4 adjacent tokens
            picked_idxs = [ai] + [n[0] for n in neighbors]
            picked_idxs.sort()
            texts = [tokens[i].text for i in picked_idxs]
            for w in range(1, min(5, len(texts)) + 1):
                for start in range(len(texts) - w + 1):
                    phrase = " ".join(texts[start : start + w]).strip()
                    phrase = self._strip_anchor_prefix(phrase)
                    if len(phrase) >= 3:
                        candidates.append((phrase, picked_idxs[start : start + w]))

        # 1b) Top-of-page candidates — for anchorless layouts, the dealer name
        #     is often in the letterhead / header region.
        if tokens:
            # Sort by vertical position; take the topmost 30%
            by_y = sorted(range(len(tokens)), key=lambda i: tokens[i].bbox[1])
            top_slice = by_y[: max(5, len(tokens) // 3)]
            top_slice.sort()  # preserve reading order
            for span in range(2, min(6, len(top_slice) + 1)):
                for s in range(len(top_slice) - span + 1):
                    idxs = top_slice[s : s + span]
                    # Only glue together tokens that are roughly same-line
                    ys = [tokens[i].bbox[1] for i in idxs]
                    if max(ys) - min(ys) > 60:  # not same line
                        continue
                    phrase = " ".join(tokens[i].text for i in idxs)
                    if len(phrase) >= 5:
                        candidates.append((phrase, idxs))

        # 2) Layout-free fallback: contiguous n-grams across whole doc.
        # Only used if we have no anchor-driven candidates at all.
        layout_free = not candidates
        if layout_free:
            for w in range(2, 5):
                for i in range(len(tokens) - w + 1):
                    phrase = " ".join(t.text for t in tokens[i : i + w])
                    candidates.append((phrase, list(range(i, i + w))))

        # 3) Score each candidate against master (using canonicalized forms)
        best_value: Optional[str] = None
        best_score: float = 0.0
        best_idxs: List[int] = []
        canon_keys = list(self._canon_to_original.keys())
        for cand, idxs in candidates:
            if not canon_keys:
                # No master — just return the highest-confidence anchor neighbor
                if len(cand) > len(best_value or ""):
                    best_value = cand
                    best_score = 75.0  # unverified match
                    best_idxs = idxs
                continue
            cand_canon = _canonicalize(cand)
            if not cand_canon or len(cand_canon) < 4:
                continue
            match_key, score, _ = process.extractOne(
                cand_canon, canon_keys, scorer=fuzz.WRatio
            )
            if score > best_score:
                best_score = float(score)
                best_value = self._canon_to_original[match_key]
                best_idxs = idxs

        # 4) Apply minimum-score gate. In layout-free fallback mode we require
        # a much higher score to avoid picking up unrelated dealer names that
        # happened to have a few common letters with some token in the doc.
        gate = max(self.threshold, 95) if layout_free else self.threshold
        if best_score >= gate:
            return ExtractionResult(best_value, best_score / 100.0, best_idxs)
        # Below the gate → return the raw candidate text, not a master entry,
        # so the downstream eval can correctly grade this as a miss rather
        # than a wrong match to the asset master.
        if best_value is not None and not layout_free:
            return ExtractionResult(best_value, best_score / 100.0, best_idxs)
        return ExtractionResult(None, 0.0, [])

    @staticmethod
    def _strip_anchor_prefix(phrase: str) -> str:
        """Remove leading 'Dealer:' / 'M/s' prefixes from a candidate."""
        p = phrase.strip()
        for prefix in ("dealer name:", "dealer:", "dealer", "m/s", "messrs"):
            if p.lower().startswith(prefix):
                p = p[len(prefix) :].strip(" :-")
        return p
