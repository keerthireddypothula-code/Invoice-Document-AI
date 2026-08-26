# Architecture

## High-level flow

```
           ┌───────────────────┐
           │  PDF / Image      │
           └────────┬──────────┘
                    │
           ┌────────▼─────────┐
           │ Ingestion        │  PyMuPDF → 300 DPI page images
           │ (src/ingest)     │  + deskew (conservative, ±1°–15°)
           │                  │  + denoise (bilateral, preserves text)
           └────────┬─────────┘
                    │
    ┌───────────────┼─────────────────┐
    │               │                 │
    ▼               ▼                 ▼
┌───────────┐  ┌─────────────┐  ┌─────────────┐
│   OCR     │  │ Detection   │  │ (page 1     │
│ PaddleOCR │  │ YOLOv8      │  │  only for   │
│ or        │  │ + heuristic │  │  detection) │
│ Tesseract │  │ fallback    │  │             │
│ multiling │  │             │  │             │
└─────┬─────┘  └──────┬──────┘  └─────────────┘
      │               │
      │               │ sig / stamp bboxes
      ▼               │ + presence + confidence
┌────────────────┐    │
│ Extractors     │    │
│ - Dealer       │    │
│   (anchor +    │    │
│    header +    │    │
│    fuzzy 90%)  │    │
│ - Model        │    │
│   (normalized  │    │
│    substring)  │    │
│ - HP           │    │
│   (regex +     │    │
│    OCR-confuse │    │
│    fallback)   │    │
│ - Cost         │    │
│   (priority-   │    │
│    weighted    │    │
│    anchors)    │    │
└───────┬────────┘    │
        │             │
        └──────┬──────┘
               │
      ┌────────▼─────────┐
      │ Fusion + Schema  │  weakest-link confidence
      │ validation       │  (absent-but-correct = 1.0)
      └────────┬─────────┘
               │
      ┌────────▼─────────┐
      │ JSON output      │  + cost + latency metrics
      └──────────────────┘
```

## Why this structure

### Separate branches for text and vision

Four of the six fields are textual (dealer, model, HP, cost) and two are
visual (signature, stamp). Splitting them lets each side use the right tools:
OCR + regex/fuzzy for text, detector + heuristics for vision. The pipeline
fuses at the JSON-assembly step.

### PaddleOCR primary, Tesseract fallback

PaddleOCR handles English + Hindi + other Indic scripts with good accuracy,
and is the right engine for multilingual processing requirements. But it
needs ~300 MB of model weights, which makes CI and Docker images heavy.
Tesseract is the fallback so the pipeline runs anywhere without network
access — with a small accuracy hit on tricky digits/scripts.

### Heuristic fallback for sig/stamp

Without labeled training data initially, we cannot reliably train
YOLO from scratch. The detector ships with a colour-+morphology heuristic
that works on any invoice out of the box. When YOLO weights are available
(`models/sig_stamp_yolov8n.pt`), it takes over as the primary detector and
heuristics become the fallback for low-confidence cases.

### Confidence fusion: weakest link

Document-level confidence is `min()` of per-field confidences. This is
conservative: one uncertain field makes the whole document uncertain, which
is exactly what we want for a human-in-the-loop review queue. We special-case
signature and stamp: if the detector confidently says "absent" and the GT is
"absent," that's a correct prediction at confidence 1.0 — we don't let the
trivial "absent" zero drag the whole doc's score.

### Pseudo-labeling for ground-truth-free training

We address the initial "no labels" constraint with confidence-bucketed pseudo-labels:
high-confidence predictions (≥0.9 on text fields, ≥0.5 on vision) are auto-
accepted as silver truth; everything else goes to the Streamlit review tab
for human correction. Corrected labels become gold truth. This implements
a practical active learning and pseudo-labeling pipeline.

## File map

| Module | Responsibility |
|---|---|
| `src/ingest/pdf_loader.py` | PDF → RGB numpy arrays (PyMuPDF or pdf2image) |
| `src/ingest/preprocess.py` | deskew, denoise, grayscale |
| `src/ocr/paddle_ocr.py` | PaddleOCR wrapper, multilingual, lazy-loaded |
| `src/ocr/tesseract_ocr.py` | Tesseract fallback, same OCRToken schema |
| `src/detection/sig_stamp_detector.py` | YOLO + heuristic orchestration |
| `src/detection/heuristics.py` | HSV mask + contour → bbox, NMS merge |
| `src/extractors/dealer_extractor.py` | anchor + header + fuzzy-to-master |
| `src/extractors/model_extractor.py` | normalized substring match |
| `src/extractors/hp_extractor.py` | regex + OCR-confusion fallback |
| `src/extractors/cost_extractor.py` | priority-weighted anchor matching |
| `src/evaluation/metrics.py` | IoU, mAP@0.5–0.95, DLA, field-level |
| `src/evaluation/evaluator.py` | full report over a batch |
| `src/data/pseudo_labeler.py` | high/low confidence bucketing |
| `src/schemas/output.py` | pydantic JSON contracts |
| `src/pipeline.py` | end-to-end orchestrator |
| `api/main.py` | FastAPI: /extract, /extract/batch |
| `app/streamlit_app.py` | demo + review UI |
| `executable.py` | standalone CLI entrypoint |

## Cost & latency

At 300 DPI + Tesseract on CPU (AWS `t3.large` pricing):

- **Latency**: ~1.9 s/doc single-page invoice
- **Cost**: ~$5.6×10⁻⁵/doc — comfortably under the $0.01 target
- **Memory peak**: ~500 MB (dominated by the page raster + OCR model)

Switching to PaddleOCR adds ~0.5 s/doc on CPU and lifts multilingual accuracy
meaningfully; still under all targets.
