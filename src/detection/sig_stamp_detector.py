"""
Signature & stamp detector.

Primary: YOLOv8 (Ultralytics) trained on signature + stamp classes.
Fallback: colour + morphology heuristics (see heuristics.py).

The detector picks the highest-confidence candidate per class. If YOLO is
below the confidence threshold AND heuristics produce a plausible candidate,
we use the heuristic. This gives us graceful degradation when training data
is scarce (which matches the problem statement: "no pre-labelled data is
provided").
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.detection.heuristics import DetectionCandidate, detect_with_heuristics
from src.schemas.output import PresenceField
from src.utils.logging import log


class SigStampDetector:
    """Unified signature + stamp detector."""

    CLASSES = ("signature", "stamp")

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        use_heuristic_fallback: bool = True,
    ):
        self.conf_threshold = conf_threshold
        self.use_heuristic_fallback = use_heuristic_fallback
        self._yolo = None
        if weights_path and Path(weights_path).exists():
            self._load_yolo(weights_path)
        else:
            log.info(
                f"YOLO weights not found at {weights_path}; "
                f"running in heuristic-only mode."
            )

    def _load_yolo(self, weights_path: str):
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(weights_path)
            log.info(f"Loaded YOLOv8 weights from {weights_path}")
        except Exception as e:
            log.warning(f"Failed to load YOLO ({e}); falling back to heuristics.")
            self._yolo = None

    def detect(self, image: np.ndarray) -> Dict[str, PresenceField]:
        """Return {'signature': PresenceField, 'stamp': PresenceField}."""
        yolo_cands = self._run_yolo(image) if self._yolo is not None else []
        heuristic_cands = (
            detect_with_heuristics(image) if self.use_heuristic_fallback else []
        )

        out: Dict[str, PresenceField] = {}
        for cls in self.CLASSES:
            cls_cands = [c for c in yolo_cands if c.label == cls]
            if not cls_cands and heuristic_cands:
                cls_cands = [c for c in heuristic_cands if c.label == cls]
            out[cls] = self._best_candidate(cls_cands)
        return out

    def _run_yolo(self, image: np.ndarray) -> List[DetectionCandidate]:
        """Run YOLO and convert to DetectionCandidate list."""
        results = self._yolo(image, conf=self.conf_threshold, verbose=False)
        cands: List[DetectionCandidate] = []
        names = self._yolo.names if hasattr(self._yolo, "names") else {}
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls.item())
                label = names.get(cls_id, str(cls_id)).lower()
                if label not in self.CLASSES:
                    continue
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf.item())
                cands.append(
                    DetectionCandidate(
                        bbox=tuple(xyxy), label=label, confidence=conf
                    )
                )
        return cands

    @staticmethod
    def _best_candidate(cands: List[DetectionCandidate]) -> PresenceField:
        if not cands:
            return PresenceField(present=False, bbox=None, confidence=0.0)
        best = max(cands, key=lambda c: c.confidence)
        return PresenceField(
            present=True,
            bbox=list(best.bbox),
            confidence=best.confidence,
        )
