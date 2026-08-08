"""
Dataset loader — iterates over invoice files (PDF or image) in a directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

SUPPORTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class InvoiceDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

    def files(self) -> List[Path]:
        """Return all supported invoice files in the root directory."""
        return sorted(
            [p for p in self.root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
        )

    def __iter__(self) -> Iterator[Path]:
        yield from self.files()

    def __len__(self) -> int:
        return len(self.files())
