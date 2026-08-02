"""
Heuristic signature & stamp detection — fallback for when YOLO is unavailable
or low-confidence.

Idea:
    - Stamps: typically circular/rectangular regions with high ink density,
      often colored (blue/red/purple) or with distinct boundaries.
    - Signatures: low-density, high-aspect-ratio handwritten strokes,
      usually in the bottom quarter of the page.

We extract connected components, filter by geometric heuristics, and score
each candidate. This isn't as good as a trained detector, but it's a
graceful fallback that keeps the system working on any invoice without
needing labeled training data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class DetectionCandidate:
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    label: str                         # 'signature' | 'stamp'
    confidence: float


def _detect_stamp_candidates(img: np.ndarray) -> List[DetectionCandidate]:
    """Detect stamp candidates using colour + circular/rect heuristics.

    Stamps are usually drawn as bright, saturated colored outlines.
    Signatures are typically dark navy/black ink — much lower V (value).
    We require both high S (saturation) AND high V to exclude signatures
    from the stamp mask.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # V >= 100 keeps bright stamp colors and rejects dark ink of signatures.
    masks = []
    masks.append(cv2.inRange(hsv, np.array([100, 80, 100]), np.array([140, 255, 255])))  # blue
    masks.append(cv2.inRange(hsv, np.array([0,   80, 100]), np.array([10,  255, 255])))  # red low
    masks.append(cv2.inRange(hsv, np.array([170, 80, 100]), np.array([180, 255, 255])))  # red high
    masks.append(cv2.inRange(hsv, np.array([130, 60, 100]), np.array([170, 255, 255])))  # purple
    base_mask = np.zeros_like(masks[0])
    for m in masks:
        base_mask = cv2.bitwise_or(base_mask, m)

    # Close the ring gap so outline becomes solid; keep kernel small enough
    # that we don't accidentally bridge to other page elements or inflate
    # the bounding box beyond the actual stamp edges.
    filled = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = img.shape[:2]
    page_area = H * W
    min_bbox_area = page_area * 0.0005
    max_bbox_area = page_area * 0.04    # stamps don't span > 4% of page bbox-wise

    candidates: List[DetectionCandidate] = []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if r < 20:
            continue
        # Use minEnclosingCircle center but size the bbox from contour's own
        # boundingRect instead of the circle radius — this hugs the stamp
        # outline rather than over-inflating.
        bx, by, bw, bh = cv2.boundingRect(cnt)
        bbox_area = bw * bh
        if bbox_area < min_bbox_area or bbox_area > max_bbox_area:
            continue
        aspect = bw / max(bh, 1)
        if not (0.7 <= aspect <= 1.4):
            continue

        # Tiny outward padding to include any anti-aliased edges.
        pad = max(3, int(min(bw, bh) * 0.03))
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(W, bx + bw + pad)
        y2 = min(H, by + bh + pad)

        conf = min(0.9, 0.5 + bbox_area / max_bbox_area)
        candidates.append(
            DetectionCandidate(
                bbox=(x1, y1, x2, y2),
                label="stamp",
                confidence=float(conf),
            )
        )

    # Merge candidates whose bboxes overlap significantly — two concentric
    # rings of the same stamp will show up as two candidates otherwise.
    candidates = _merge_overlapping(candidates)
    return candidates


def _merge_overlapping(cands: List[DetectionCandidate]) -> List[DetectionCandidate]:
    """Greedily merge candidates whose bboxes overlap > 30% (IoU)."""
    if not cands:
        return cands

    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # Sort by area descending; merge smaller boxes into larger neighbors.
    ordered = sorted(
        cands,
        key=lambda c: (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1]),
        reverse=True,
    )
    merged: List[DetectionCandidate] = []
    for c in ordered:
        joined = False
        for i, m in enumerate(merged):
            if _iou(c.bbox, m.bbox) > 0.3:
                # Union of boxes, max of confidences
                nx1 = min(c.bbox[0], m.bbox[0])
                ny1 = min(c.bbox[1], m.bbox[1])
                nx2 = max(c.bbox[2], m.bbox[2])
                ny2 = max(c.bbox[3], m.bbox[3])
                merged[i] = DetectionCandidate(
                    bbox=(nx1, ny1, nx2, ny2),
                    label=m.label,
                    confidence=max(m.confidence, c.confidence),
                )
                joined = True
                break
        if not joined:
            merged.append(c)
    return merged


def _detect_signature_candidates(img: np.ndarray) -> List[DetectionCandidate]:
    """Detect signature candidates — dense strokes, bottom of page."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    H, W = binary.shape
    lower = binary[H // 2 :, :].copy()

    # Dilate to merge strokes of a single signature into one blob
    kernel = np.ones((5, 15), np.uint8)
    dil = cv2.dilate(lower, kernel, iterations=2)

    num, _, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)

    candidates: List[DetectionCandidate] = []
    page_area = H * W
    min_area = page_area * 0.001
    max_area = page_area * 0.05

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        aspect = w / max(h, 1)
        if not (1.5 <= aspect <= 10):
            continue

        # Recover tight bbox from the un-dilated strokes inside this component.
        sub = lower[y : y + h, x : x + w]
        ys_, xs_ = np.where(sub > 0)
        if len(ys_) < 30:
            continue
        tight_x1 = int(xs_.min()) + x
        tight_y1 = int(ys_.min()) + y
        tight_x2 = int(xs_.max()) + x + 1
        tight_y2 = int(ys_.max()) + y + 1
        pad = max(3, int(min(w, h) * 0.05))

        conf = min(0.85, 0.35 + area / max_area)
        candidates.append(
            DetectionCandidate(
                bbox=(
                    max(0, tight_x1 - pad),
                    max(0, tight_y1 + H // 2 - pad),       # shift back to full image
                    min(W, tight_x2 + pad),
                    min(H, tight_y2 + H // 2 + pad),
                ),
                label="signature",
                confidence=float(conf),
            )
        )
    return candidates


def detect_with_heuristics(image: np.ndarray) -> List[DetectionCandidate]:
    """Run both signature and stamp heuristics on an image."""
    sigs = _detect_signature_candidates(image)
    stamps = _detect_stamp_candidates(image)
    return sigs + stamps
