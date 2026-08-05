"""
InvoicePipeline — the end-to-end orchestrator.

Flow:
    1. Load PDF → page images
    2. Preprocess (deskew, denoise)
    3. OCR (PaddleOCR, multilingual)
    4. Detect signature + stamp (YOLOv8 + heuristic fallback)
    5. Extract textual fields (dealer, model, HP, cost)
    6. Fuse confidences
    7. Emit DocumentOutput JSON
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.detection.sig_stamp_detector import SigStampDetector
from src.extractors.cost_extractor import CostExtractor
from src.extractors.dealer_extractor import DealerExtractor
from src.extractors.hp_extractor import HPExtractor
from src.extractors.model_extractor import ModelExtractor
from src.ingest.pdf_loader import load_image, load_pdf_as_images
from src.ingest.preprocess import preprocess
from src.ocr.paddle_ocr import PaddleOCREngine
from src.schemas.output import (
    DocumentOutput,
    ExtractedFields,
    FieldConfidences,
    OCRToken,
    PresenceField,
)
from src.utils.cost_tracker import estimate_cost_usd
from src.utils.io import load_config
from src.utils.logging import log


class InvoicePipeline:
    def __init__(self, config_path: str = "configs/default.yaml", cfg: Optional[dict] = None):
        self.cfg = cfg if cfg is not None else load_config(config_path)

        # Lazy-init heavy components so tests can mock them.
        self._ocr: Optional[PaddleOCREngine] = None
        self._detector: Optional[SigStampDetector] = None

        # Extractors are cheap — instantiate eagerly
        self.dealer_ex = DealerExtractor(self.cfg)
        self.model_ex = ModelExtractor(self.cfg)
        self.hp_ex = HPExtractor(self.cfg)
        self.cost_ex = CostExtractor(self.cfg)

    # --- lazy accessors ---------------------------------------------------

    @property
    def ocr(self):
        if self._ocr is None:
            ocr_cfg = self.cfg.get("ocr", {})
            engine = ocr_cfg.get("engine", "paddleocr").lower()
            if engine == "tesseract":
                from src.ocr.tesseract_ocr import TesseractEngine
                self._ocr = TesseractEngine(
                    lang=ocr_cfg.get("tesseract_lang", "eng"),
                    min_confidence=ocr_cfg.get("min_confidence", 0.3),
                )
            else:
                try:
                    self._ocr = PaddleOCREngine(
                        languages=ocr_cfg.get("languages", ["en", "hi"]),
                        use_angle_cls=ocr_cfg.get("use_angle_cls", True),
                        min_confidence=ocr_cfg.get("min_confidence", 0.5),
                    )
                except ImportError:
                    log.warning("PaddleOCR not installed; falling back to Tesseract.")
                    from src.ocr.tesseract_ocr import TesseractEngine
                    self._ocr = TesseractEngine(
                        lang=ocr_cfg.get("tesseract_lang", "eng"),
                        min_confidence=ocr_cfg.get("min_confidence", 0.3),
                    )
        return self._ocr

    @property
    def detector(self) -> SigStampDetector:
        if self._detector is None:
            det_cfg = self.cfg.get("detection", {})
            self._detector = SigStampDetector(
                weights_path=self.cfg.get("paths", {}).get("yolo_weights"),
                conf_threshold=det_cfg.get("yolo_conf_threshold", 0.35),
                use_heuristic_fallback=det_cfg.get("heuristic_fallback", True),
            )
        return self._detector

    # --- main entry point -------------------------------------------------

    def run(self, input_path: str, doc_id: Optional[str] = None) -> DocumentOutput:
        """Process one document and return a DocumentOutput."""
        t0 = time.time()
        path = Path(input_path)
        doc_id = doc_id or path.stem

        # 1. Ingestion — support both PDF and single images
        if path.suffix.lower() == ".pdf":
            pages = load_pdf_as_images(
                path, dpi=self.cfg.get("ingest", {}).get("dpi", 300)
            )
        else:
            pages = [load_image(path)]

        log.info(f"[{doc_id}] loaded {len(pages)} page(s)")

        # 2. Preprocess each page
        ing = self.cfg.get("ingest", {})
        pages = [
            preprocess(p, do_deskew=ing.get("deskew", True), do_denoise=ing.get("denoise", True))
            for p in pages
        ]

        # 3. OCR — collect tokens across pages
        all_tokens: List[OCRToken] = []
        for idx, page_img in enumerate(pages):
            try:
                tokens = self.ocr.run(page_img, page=idx)
            except Exception as e:
                log.error(f"[{doc_id}] OCR failed on page {idx}: {e}")
                tokens = []
            all_tokens.extend(tokens)
        log.debug(f"[{doc_id}] extracted {len(all_tokens)} OCR tokens")

        # 4. Detection — run on first page (signatures/stamps are typically there)
        try:
            det = self.detector.detect(pages[0])
        except Exception as e:
            log.error(f"[{doc_id}] detection failed: {e}")
            det = {
                "signature": PresenceField(present=False),
                "stamp": PresenceField(present=False),
            }

        # 5. Field extraction
        dealer = self.dealer_ex.extract(all_tokens)
        model = self.model_ex.extract(all_tokens)
        hp = self.hp_ex.extract(all_tokens)
        cost = self.cost_ex.extract(all_tokens)

        fields = ExtractedFields(
            dealer_name=dealer.value,
            model_name=model.value,
            horse_power=hp.value if isinstance(hp.value, int) else None,
            asset_cost=cost.value if isinstance(cost.value, int) else None,
            signature=det["signature"],
            stamp=det["stamp"],
        )

        confs = FieldConfidences(
            dealer_name=float(dealer.confidence),
            model_name=float(model.confidence),
            horse_power=float(hp.confidence),
            asset_cost=float(cost.confidence),
            signature=float(det["signature"].confidence),
            stamp=float(det["stamp"].confidence),
        )

        # Document-level confidence = min over fields (weakest-link principle).
        # For signature / stamp, an absent-but-detected-as-absent field is a
        # confident "no" — don't let it drag down the document score.
        sig_contrib = confs.signature if det["signature"].present else 1.0
        stamp_contrib = confs.stamp if det["stamp"].present else 1.0
        doc_conf = min(
            confs.dealer_name,
            confs.model_name,
            confs.horse_power,
            confs.asset_cost,
            sig_contrib,
            stamp_contrib,
        )

        elapsed = time.time() - t0
        out = DocumentOutput(
            doc_id=doc_id,
            fields=fields,
            field_confidences=confs,
            confidence=round(float(doc_conf), 3),
            processing_time_sec=round(elapsed, 3),
            cost_estimate_usd=estimate_cost_usd(elapsed),
            num_pages=len(pages),
        )
        log.info(
            f"[{doc_id}] done in {elapsed:.2f}s | conf={doc_conf:.2f} | "
            f"dealer={fields.dealer_name} model={fields.model_name} "
            f"hp={fields.horse_power} cost={fields.asset_cost}"
        )
        return out
