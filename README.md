# Intelligent Document AI for Invoice Field Extraction

An end-to-end Document AI pipeline that ingests PDF / image invoices
(tractor loan quotations and generic retail invoices), extracts six
structured fields, and emits a JSON with bounding boxes and confidence
scores — designed for multilingual Indian documents (English / Hindi /
Gujarati), robust across digital, scanned and handwritten layouts.

## TL;DR

- **500+** synthetic invoices generated across 3 layouts × 3 quality levels × multilingual anchors
- **6 fields** extracted per document: dealer name, model, HP, asset cost, signature, stamp
- **Multilingual OCR** with PaddleOCR (en / hi) and a Tesseract fallback
- **Computer Vision + NLP** split: colour-and-morphology sig/stamp detector, anchor-and-fuzzy field extractors, fused under a weakest-link confidence model
- **Signature and stamp detection** with bounding boxes validated at **IoU > 0.5**
- End-to-end FastAPI service, Streamlit demo + human-in-the-loop review UI, full pytest suite (18/18 passing), Dockerfile

---

## Measured results (50-document validation set, Tesseract fallback OCR)

| Metric | Value | Target |
|---|---|---|
| **Document-Level Accuracy (all 6 fields correct)** | **52%** | 95% (with full PaddleOCR + labeled training data) |
| Dealer name (fuzzy ≥ 90% vs master) | **82%** | — |
| Model name (exact substring match on normalized form) | **84%** | — |
| Horse Power (exact, ±5% tolerance) | **96%** | — |
| Asset Cost (exact, ±5% tolerance) | **94%** | — |
| Signature presence + IoU > 0.5 | **80%** | — |
| Stamp presence + IoU > 0.5 | **64%** | — |
| Signature mAP @ 0.5–0.95 | **0.37** | — |
| Stamp mAP @ 0.5–0.95 | **0.15** | — |
| Avg latency | **1.9 s/doc** | ≤ 30 s |
| Estimated cost | **~$5.6 × 10⁻⁵/doc** | < $0.01 |

The validation set is 50 auto-generated invoices with perfect ground truth,
run on CPU with the Tesseract fallback engine (PaddleOCR wasn't installed
in the CI container for size reasons). Field-level accuracy on the textual
fields is already 82–96%; sig/stamp bbox accuracy lifts substantially when
a YOLOv8 detector is trained on even a small (~50-doc) labeled set — the
hook is already in `src/detection/sig_stamp_detector.py`, just drop weights
into `models/sig_stamp_yolov8n.pt`.

---

## System architecture

See [`docs/architecture.md`](docs/architecture.md) for the full diagram.
At a glance:

```
PDF/Image  ──▶  Ingest (PyMuPDF + deskew + denoise)
               │
               ├──▶  OCR (PaddleOCR multilingual / Tesseract fallback)
               │       │
               │       └──▶  Extractors
               │              ├─ Dealer   (anchor + header fallback + fuzzy ≥ 90% vs master)
               │              ├─ Model    (normalized substring match)
               │              ├─ HP       (regex + OCR-confusion fallback: S→5 etc.)
               │              └─ Cost     (priority-weighted anchors + range gate)
               │
               └──▶  Detector (YOLOv8 + heuristic fallback: HSV + contours + NMS)
                       │
                       └──▶  signature & stamp bboxes + confidences
                                │
                                ▼
                       Fusion + schema validation
                                │
                                ▼
                          JSON output
```

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate 500 synthetic invoices (optional — for testing)
python scripts/generate_sample_data.py --out data/synthetic --n 500

# 3. Run on a single file
python executable.py --input path/to/invoice.pdf --output result.json

# 4. Run on a folder
python executable.py --input data/synthetic/images --output outputs/predictions

# 5. Evaluate against ground truth
PYTHONPATH=. python scripts/evaluate.py \
    --predictions outputs/predictions \
    --ground_truth data/synthetic/ground_truth \
    --report outputs/eval_report.json

# 6. Launch demo UI
streamlit run app/streamlit_app.py

# 7. Launch API server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Project structure

```
invoice-document-ai/
├── executable.py              # standalone CLI entrypoint
├── requirements.txt
├── Dockerfile
├── configs/
│   ├── default.yaml           # thresholds, anchors, paths
│   └── asset_master.csv       # dealer / model reference data
├── src/
│   ├── pipeline.py            # end-to-end orchestrator
│   ├── ingest/                # PDF loading + preprocessing
│   ├── ocr/                   # PaddleOCR + Tesseract engines
│   ├── detection/             # YOLOv8 + heuristic sig/stamp detector
│   ├── extractors/            # 4 field extractors (dealer/model/hp/cost)
│   ├── evaluation/            # IoU, mAP, DLA metrics + evaluator
│   ├── data/                  # dataset loader + pseudo-labeler
│   ├── schemas/               # pydantic output contracts
│   └── utils/                 # io, logging, cost tracker
├── api/main.py                # FastAPI: /extract, /extract/batch, /schema
├── app/streamlit_app.py       # Streamlit demo + HITL review
├── scripts/
│   ├── generate_sample_data.py  # synthetic invoice factory
│   ├── run_batch.py             # batch pipeline
│   └── evaluate.py              # eval harness
├── tests/                     # 18 unit tests covering metrics + extractors
├── sample_output/result.json  # example prediction in spec format
└── docs/architecture.md       # full diagram + design notes
```

---

## Key design choices

**PaddleOCR primary, Tesseract fallback.** PaddleOCR covers English + Hindi
+ other Indic scripts with solid accuracy. We lazy-load it so tests and
Streamlit cold-starts stay fast. Tesseract is a lightweight fallback so the
pipeline runs in any container without downloading ~300 MB of model weights.

**Dealer extractor does more than fuzzy matching.** Three layers: (1) anchor
tokens like "Dealer:" / "M/s" / Hindi "डीलर" with right-of and below
spatial association; (2) if anchorless, top-of-page header tokens joined
same-line; (3) fuzzy-matched against a canonicalized master list where
"Pvt Ltd" ↔ "Private Limited" don't count as a mismatch. A minimum-score
gate prevents false matches when the real dealer line isn't on the page.

**HP extractor handles OCR noise.** Two regex passes: a strict `\d+ HP`
pattern first, then a permissive pattern that accepts common OCR
confusions (`S→5, O→0, I→1, B→8`) when the strict pass finds nothing.
Catches cases like Tesseract reading "55 HP" as "SS HP".

**Stamp detector under no labels.** HSV colour mask (blue/red/purple
with V≥100 to exclude dark signature ink), morphology to fill circular
outlines, contour bounding rect, NMS-merge of concentric rings. Graceful
degradation path: drop a YOLOv8 weights file into `models/` and the
detector automatically prefers it.

**Pseudo-labeling workflow.** `src/data/pseudo_labeler.py` partitions
predictions by confidence: high-confidence become silver labels
(auto-accepted), low-confidence go to the Streamlit review tab where a
human corrects and saves them as gold, creating a continuous improvement loop through active learning and pseudo-labeling.

**Confidence fusion: weakest link.** Document-level confidence is the min
of field confidences, with one special case: signature/stamp confidently
detected as *absent* contributes 1.0 (not 0.0) so trivial absences don't
tank the score.

---

## Running tests

```bash
pytest tests/ -v
# 18 passed
```

---

## Extending to real data

Drop real PDF invoices into `data/real/` and run:

```bash
python executable.py --input data/real --output outputs/real_predictions
```

If you have partial ground truth, label a sample in the same JSON format
as `data/synthetic/ground_truth/` and run the evaluator. For signature/
stamp training: label ~50 invoices with LabelImg or Roboflow, export in
YOLO format, fine-tune `yolov8n.pt` for 50 epochs, drop the weights into
`models/sig_stamp_yolov8n.pt` — the detector will auto-prefer it.

---

## License

MIT.
