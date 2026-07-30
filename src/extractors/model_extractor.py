"""
Model name extractor — exact match against asset master's `model_name` column.

Because the problem statement requires *exact* match for model name, we do:
    1. Find anchor tokens ("model", "model no", etc.).
    2. Greedy longest-substring match of surrounding text against each known
       model in the master. This tolerates OCR noise ("Mahindra575DI" vs
       "Mahindra 575 DI") via normalized comparison, but the returned value
       is always the canonical form from the master.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
from rapidfuzz import fuzz, process

from src.extractors.base import BaseExtractor, ExtractionResult
from src.schemas.output import OCRToken


def _normalize(s: str) -> str:
    return re.sub(r"[\s\-_/.]+", "", s).lower()


class ModelExtractor(BaseExtractor):
    field_name = "model_name"

    def __init__(self, cfg: dict, asset_master_path: Optional[str] = None):
        super().__init__(cfg)
        ext_cfg = cfg.get("extraction", {}).get("model", {})
        self.anchors: List[str] = ext_cfg.get("anchors", ["model"])

        master_path = asset_master_path or cfg.get("paths", {}).get(
            "asset_master", "configs/asset_master.csv"
        )
        self.model_list: List[str] = []
        self.model_norm_to_canonical: dict[str, str] = {}
        if Path(master_path).exists():
            df = pd.read_csv(master_path)
            for m in df["model_name"].dropna().unique().tolist():
                self.model_list.append(m)
                self.model_norm_to_canonical[_normalize(m)] = m

    def extract(self, tokens: List[OCRToken]) -> ExtractionResult:
        if not tokens or not self.model_list:
            return ExtractionResult(None, 0.0, [])

        # Join everything to one giant string (normalized) — substring search
        joined = " ".join(t.text for t in tokens)
        joined_norm = _normalize(joined)

        # Exact substring of normalized model name in normalized doc text
        best_value: Optional[str] = None
        best_score = 0.0
        for norm_m, canonical in self.model_norm_to_canonical.items():
            if norm_m in joined_norm:
                # Exact match — high confidence
                if len(canonical) > len(best_value or ""):
                    best_value = canonical
                    best_score = 0.98
        if best_value is not None:
            return ExtractionResult(best_value, best_score, [])

        # Otherwise: anchor-driven fuzzy scoring
        anchor_idxs = self._find_anchor_indices(tokens, self.anchors)
        candidates: List[str] = []
        for ai in anchor_idxs:
            neighbors = self._neighbors_right_or_below(tokens, ai, max_candidates=5)
            picked = sorted([ai] + [n[0] for n in neighbors])
            texts = [tokens[i].text for i in picked]
            for w in range(1, min(4, len(texts)) + 1):
                for start in range(len(texts) - w + 1):
                    cand = " ".join(texts[start : start + w])
                    candidates.append(cand)

        for cand in candidates:
            match, score, _ = process.extractOne(
                cand, self.model_list, scorer=fuzz.WRatio
            )
            if score > best_score * 100 and score >= 85:
                best_value = match
                best_score = score / 100.0

        return ExtractionResult(best_value, best_score, [])
