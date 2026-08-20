"""
executable.py — standalone CLI entry point for batch and single-file invoice processing.

Usage:
    # Process a single PDF
    python executable.py --input path/to/invoice.pdf --output result.json

    # Process a directory of PDFs, write one JSON per input
    python executable.py --input path/to/dir --output outputs/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.dataset import InvoiceDataset
from src.pipeline import InvoicePipeline


def _process_one(pipe: InvoicePipeline, input_path: Path, output_path: Path):
    result = pipe.run(str(input_path), doc_id=input_path.stem)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"[ok] {input_path.name} → {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Invoice Document AI — main runner")
    ap.add_argument("--input", required=True, help="Path to PDF/image or a directory")
    ap.add_argument("--output", required=True, help="Output JSON path or directory")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    pipe = InvoicePipeline(config_path=args.config)
    in_path = Path(args.input)
    out_path = Path(args.output)

    if in_path.is_file():
        # Single input
        if out_path.is_dir() or args.output.endswith("/"):
            out_file = out_path / f"{in_path.stem}.json"
        else:
            out_file = out_path
        _process_one(pipe, in_path, out_file)
    elif in_path.is_dir():
        ds = InvoiceDataset(in_path)
        for f in ds.files():
            _process_one(pipe, f, out_path / f"{f.stem}.json")
    else:
        raise FileNotFoundError(f"--input not found: {in_path}")


if __name__ == "__main__":
    main()
