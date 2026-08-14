"""
FastAPI app exposing the pipeline over HTTP.

Endpoints:
    GET  /health                — liveness probe
    POST /extract               — upload a PDF/image, get JSON extraction
    POST /extract/batch         — upload multiple files
    GET  /schema                — returns the output JSON schema
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.pipeline import InvoicePipeline
from src.schemas.output import DocumentOutput
from src.utils.logging import log


app = FastAPI(
    title="Invoice Document AI",
    description=(
        "Multilingual, CV+NLP pipeline for extracting structured fields from "
        "tractor loan quotations and general invoices. Returns JSON with "
        "dealer, model, HP, cost, signature & stamp (with bboxes)."
    ),
    version="1.0.0",
)

# Pipeline is initialized lazily on first request so the app starts fast.
_pipeline: InvoicePipeline | None = None


def get_pipeline() -> InvoicePipeline:
    global _pipeline
    if _pipeline is None:
        log.info("Initializing pipeline on first request...")
        _pipeline = InvoicePipeline()
    return _pipeline


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    return DocumentOutput.model_json_schema()


@app.post("/extract", response_model=DocumentOutput)
async def extract(file: UploadFile = File(...)):
    """Extract fields from a single uploaded PDF/image."""
    suffix = Path(file.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        content = await file.read()
        tf.write(content)
        tmp_path = tf.name

    try:
        pipe = get_pipeline()
        result = pipe.run(tmp_path, doc_id=Path(file.filename).stem)
        return result
    except Exception as e:
        log.exception("extract failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/extract/batch")
async def extract_batch(files: List[UploadFile] = File(...)):
    """Extract fields from a batch of files."""
    pipe = get_pipeline()
    results = []
    for file in files:
        suffix = Path(file.filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(await file.read())
            tmp_path = tf.name
        try:
            out = pipe.run(tmp_path, doc_id=Path(file.filename).stem)
            results.append(out.model_dump())
        except Exception as e:
            results.append({"doc_id": Path(file.filename).stem, "error": str(e)})
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    return JSONResponse(content={"results": results, "count": len(results)})
