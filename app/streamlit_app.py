"""
Streamlit demo + human-in-the-loop review UI.

Two modes (tabs):
    1. "Extract"  — upload a PDF/image, see fields + overlaid bboxes
    2. "Review"   — browse low-confidence predictions, correct them,
                    save as gold labels.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.ingest.pdf_loader import load_pdf_as_images, load_image
from src.pipeline import InvoicePipeline


st.set_page_config(
    page_title="Invoice Document AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Cached pipeline ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading pipeline...")
def get_pipeline() -> InvoicePipeline:
    return InvoicePipeline()


def draw_boxes(img: np.ndarray, result_dict: dict) -> np.ndarray:
    out = img.copy()
    fields = result_dict.get("fields", {})
    colors = {"signature": (0, 200, 0), "stamp": (0, 100, 255)}
    for name in ("signature", "stamp"):
        f = fields.get(name, {})
        if f.get("present") and f.get("bbox"):
            x1, y1, x2, y2 = f["bbox"]
            color = colors[name]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                out, name, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            )
    return out


# ─── Sidebar ─────────────────────────────────────────────────────────────
st.sidebar.title("Invoice Document AI")
st.sidebar.markdown(
    "Multilingual CV+NLP pipeline for extracting structured fields from "
    "tractor loan quotations and general invoices."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Target metrics**")
st.sidebar.markdown("- DLA ≥ 95%")
st.sidebar.markdown("- IoU > 0.5 for sig/stamp")
st.sidebar.markdown("- Latency < 30s / doc")
st.sidebar.markdown("- Cost < $0.01 / doc")

tab1, tab2 = st.tabs(["Extract", "Human-in-the-loop Review"])

# ─── Tab 1: Extract ──────────────────────────────────────────────────────
with tab1:
    st.header("Upload Invoice")
    uploaded = st.file_uploader(
        "Choose a PDF or image", type=["pdf", "png", "jpg", "jpeg"]
    )
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded.name).suffix
        ) as tf:
            tf.write(uploaded.read())
            tmp_path = tf.name

        pipe = get_pipeline()
        with st.spinner("Running pipeline..."):
            result = pipe.run(tmp_path, doc_id=Path(uploaded.name).stem)
        result_dict = result.model_dump()

        # Load first page for visualization
        if Path(tmp_path).suffix.lower() == ".pdf":
            first_page = load_pdf_as_images(tmp_path, dpi=200)[0]
        else:
            first_page = load_image(tmp_path)

        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Document (with detections)")
            annotated = draw_boxes(first_page, result_dict)
            st.image(annotated, use_column_width=True)

        with c2:
            st.subheader("Extracted Fields")
            f = result_dict["fields"]
            confs = result_dict["field_confidences"]
            for key in ("dealer_name", "model_name", "horse_power", "asset_cost"):
                st.metric(
                    key.replace("_", " ").title(),
                    str(f.get(key, "—")),
                    delta=f"conf: {confs.get(key, 0):.2f}",
                )
            st.markdown("**Signature**: "
                        f"{'✓ present' if f['signature']['present'] else '✗ missing'} "
                        f"(conf {confs['signature']:.2f})")
            st.markdown("**Stamp**: "
                        f"{'✓ present' if f['stamp']['present'] else '✗ missing'} "
                        f"(conf {confs['stamp']:.2f})")
            st.markdown("---")
            st.markdown(f"**Overall confidence:** {result_dict['confidence']:.2f}")
            st.markdown(f"**Processing time:** {result_dict['processing_time_sec']:.2f}s")
            st.markdown(f"**Est. cost (CPU):** ${result_dict['cost_estimate_usd']:.5f}")

        st.subheader("Raw JSON")
        st.json(result_dict)

        st.download_button(
            "Download JSON",
            data=json.dumps(result_dict, indent=2),
            file_name=f"{Path(uploaded.name).stem}_result.json",
            mime="application/json",
        )

# ─── Tab 2: Review ───────────────────────────────────────────────────────
with tab2:
    st.header("Review Low-Confidence Predictions")
    st.markdown(
        "This workflow addresses the **'no ground truth'** setting. "
        "Low-confidence predictions are queued here for human correction, "
        "and become gold labels."
    )
    review_dir = st.text_input(
        "Review directory", "outputs/pseudo_labels/for_review"
    )
    gold_dir = st.text_input(
        "Save corrected labels to", "outputs/gold_labels"
    )

    rp = Path(review_dir)
    if not rp.exists():
        st.warning(f"Directory not found: {rp}")
    else:
        files = sorted(rp.glob("*.json"))
        if not files:
            st.success("No documents pending review.")
        else:
            choice = st.selectbox("Pick a doc to review", [f.name for f in files])
            target = rp / choice
            with open(target) as fh:
                pred = json.load(fh)

            st.subheader(f"Original prediction: {choice}")
            st.json(pred)

            st.subheader("Correct the fields")
            f = pred.get("fields", {})
            new_dealer = st.text_input("dealer_name", f.get("dealer_name") or "")
            new_model = st.text_input("model_name", f.get("model_name") or "")
            new_hp = st.number_input(
                "horse_power", value=int(f.get("horse_power") or 0), step=1
            )
            new_cost = st.number_input(
                "asset_cost", value=int(f.get("asset_cost") or 0), step=1000
            )
            sig_present = st.checkbox("signature present", value=f.get("signature", {}).get("present", False))
            stamp_present = st.checkbox("stamp present", value=f.get("stamp", {}).get("present", False))

            if st.button("Save as gold"):
                Path(gold_dir).mkdir(parents=True, exist_ok=True)
                corrected = dict(pred)
                corrected["fields"] = {
                    **f,
                    "dealer_name": new_dealer,
                    "model_name": new_model,
                    "horse_power": int(new_hp),
                    "asset_cost": int(new_cost),
                    "signature": {
                        **f.get("signature", {}),
                        "present": sig_present,
                    },
                    "stamp": {
                        **f.get("stamp", {}),
                        "present": stamp_present,
                    },
                }
                corrected["source"] = "human_review"
                out_path = Path(gold_dir) / choice
                with open(out_path, "w") as fh:
                    json.dump(corrected, fh, indent=2)
                st.success(f"Saved gold label → {out_path}")
