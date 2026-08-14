"""
Batch runner — processes a directory of invoices and writes one JSON per doc.

Usage:
    python scripts/run_batch.py \
        --input data/synthetic/images \
        --output outputs/predictions \
        --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.data.dataset import InvoiceDataset
from src.pipeline import InvoicePipeline
from src.utils.io import save_json
from src.utils.logging import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Directory of invoices")
    ap.add_argument("--output", default="outputs/predictions")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on number of docs to process")
    args = ap.parse_args()

    ds = InvoiceDataset(args.input)
    files = ds.files()
    if args.limit:
        files = files[: args.limit]

    log.info(f"Processing {len(files)} files from {args.input}")
    pipe = InvoicePipeline(config_path=args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    successes, failures = 0, 0
    total_time = 0.0
    for f in tqdm(files, desc="batch"):
        try:
            result = pipe.run(str(f), doc_id=f.stem)
            save_json(result.model_dump(), out_dir / f"{f.stem}.json")
            successes += 1
            total_time += result.processing_time_sec
        except Exception as e:
            log.error(f"Failed on {f}: {e}")
            failures += 1

    avg = total_time / max(1, successes)
    log.info(
        f"Batch done: {successes} ok, {failures} failed. "
        f"Avg latency: {avg:.2f}s/doc."
    )


if __name__ == "__main__":
    main()
