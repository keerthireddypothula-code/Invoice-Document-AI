"""
Image preprocessing for better OCR accuracy on scanned / photographed invoices.

Steps:
    1. Grayscale conversion
    2. Deskew (via minimum-area rectangle on text contours)
    3. Denoise (bilateral filter — preserves text edges)
    4. Adaptive thresholding (optional — helps scanned docs)
"""
from __future__ import annotations

import cv2
import numpy as np


def grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def deskew(img: np.ndarray) -> np.ndarray:
    """Deskew based on the dominant text orientation.

    We binarize, find coordinates of dark (text) pixels, fit a minimum-area
    rotated rectangle, and rotate by its angle. Works well for small skews
    (< 15°) typical of phone-captured invoices. Conservative — bails out if
    the estimated angle is tiny or suspiciously large.
    """
    gray = grayscale(img)
    # Invert + threshold so text pixels become foreground (non-zero)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size < 100:
        return img  # too little content; skip deskewing

    # Subsample if very large (speed + robustness)
    if coords.shape[0] > 5000:
        idx = np.random.choice(coords.shape[0], 5000, replace=False)
        coords = coords[idx]

    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV returns angle in [-90, 0); convert to [-45, 45]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only rotate for meaningful skews. Tiny angles (< 1°) are noise; large
    # angles (> 15°) usually mean the detector was fooled by the text layout.
    if abs(angle) < 1.0 or abs(angle) > 15.0:
        return img

    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def denoise(img: np.ndarray) -> np.ndarray:
    """Bilateral filter preserves sharp edges (good for text)."""
    if img.ndim == 3:
        return cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
    return cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)


def preprocess(img: np.ndarray, do_deskew: bool = True, do_denoise: bool = True) -> np.ndarray:
    """Run the full preprocessing chain."""
    out = img.copy()
    if do_deskew:
        out = deskew(out)
    if do_denoise:
        out = denoise(out)
    return out
