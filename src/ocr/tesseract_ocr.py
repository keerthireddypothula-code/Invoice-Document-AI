"""
Tesseract fallback OCR — emits the same OCRToken schema as PaddleOCR.

Rationale: PaddleOCR is the primary engine (≈ 300 MB of model weights,
best multilingual accuracy), but Tesseract is much lighter and already
ships on most Linux distros. This fallback lets the pipeline run in
constrained environments (CI runners, Docker images < 500 MB) without
changing the pipeline interface.
"""
from __future__ import annotations

from typing import List

import numpy as np

from src.schemas.output import OCRToken


class TesseractEngine:
    def __init__(self, lang: str = "eng", min_confidence: float = 0.3):
        self.lang = lang
        self.min_confidence = min_confidence

    def run(self, image: np.ndarray, page: int = 0) -> List[OCRToken]:
        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(
            image, lang=self.lang, output_type=Output.DICT
        )
        tokens: List[OCRToken] = []
        n = len(data["text"])
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i]) / 100.0
            except (ValueError, TypeError):
                conf = 0.0
            if conf < self.min_confidence:
                continue
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            tokens.append(
                OCRToken(
                    text=text,
                    bbox=[x, y, x + w, y + h],
                    confidence=conf,
                    page=page,
                )
            )

        # Also emit line-level tokens (concat of word tokens in same block/line)
        # so that multi-word values like "ABC Tractors Pvt Ltd" become one token
        # for the extractor — this greatly improves dealer extraction accuracy.
        line_tokens = self._collate_lines(data)
        return tokens + line_tokens

    @staticmethod
    def _collate_lines(data) -> List[OCRToken]:
        lines: dict = {}
        n = len(data["text"])
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            try:
                conf = float(data["conf"][i]) / 100.0
            except (ValueError, TypeError):
                conf = 0.0
            if conf < 0.3:
                continue
            x1 = int(data["left"][i])
            y1 = int(data["top"][i])
            x2 = x1 + int(data["width"][i])
            y2 = y1 + int(data["height"][i])
            if key not in lines:
                lines[key] = {"text": [text], "bbox": [x1, y1, x2, y2],
                              "confs": [conf]}
            else:
                lines[key]["text"].append(text)
                b = lines[key]["bbox"]
                lines[key]["bbox"] = [
                    min(b[0], x1), min(b[1], y1),
                    max(b[2], x2), max(b[3], y2),
                ]
                lines[key]["confs"].append(conf)

        out: List[OCRToken] = []
        for info in lines.values():
            if len(info["text"]) < 2:
                continue  # already emitted as word tokens
            out.append(OCRToken(
                text=" ".join(info["text"]),
                bbox=info["bbox"],
                confidence=sum(info["confs"]) / len(info["confs"]),
                page=0,
            ))
        return out
