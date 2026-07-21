"""
PDF → image loader.

Uses PyMuPDF (fitz) because it's pure-Python-friendly and doesn't need poppler
on the host. Falls back to pdf2image if PyMuPDF isn't available.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def load_pdf_as_images(pdf_path: str | Path, dpi: int = 300) -> List[np.ndarray]:
    """Rasterize every page of a PDF into an RGB numpy array.

    Args:
        pdf_path: path to the input PDF.
        dpi: rasterization DPI. 300 is a good tradeoff between OCR accuracy
             and memory / speed.

    Returns:
        List of HxWx3 uint8 numpy arrays (one per page).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        import fitz  # PyMuPDF
    except ImportError:
        return _load_with_pdf2image(pdf_path, dpi)

    images: List[np.ndarray] = []
    doc = fitz.open(pdf_path)
    try:
        zoom = dpi / 72.0  # 72 is the PDF base DPI
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            # PyMuPDF may give RGB or RGBA depending on alpha; normalize to RGB
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            images.append(arr.copy())
    finally:
        doc.close()

    return images


def _load_with_pdf2image(pdf_path: Path, dpi: int) -> List[np.ndarray]:
    """Fallback path using pdf2image + poppler."""
    from pdf2image import convert_from_path

    pil_pages = convert_from_path(str(pdf_path), dpi=dpi)
    return [np.array(p.convert("RGB")) for p in pil_pages]


def load_image(image_path: str | Path) -> np.ndarray:
    """Load a single image file (png/jpg/tif) as an RGB ndarray."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not readable: {image_path}")
    # cv2 loads BGR; convert to RGB for consistency with PDF path
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
