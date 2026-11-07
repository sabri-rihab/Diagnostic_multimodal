import json
import base64
from pathlib import Path

import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import io

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
MODELS     = ROOT / "models"
YOLO_DIR   = ROOT / "yolo_model_results" / "detection" / "yolov8n_v1-4"

st.set_page_config(page_title="Brain Tumor Diagnostic", layout="wide")

# API_CLS = "http://api-classification:8001"
API_CLS = "http://localhost:8001"
# API_DET = "http://api-detection:8002"
API_DET = "http://localhost:8002"

page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "🧠 Classification",
    "🔍 YOLO Detection",
    "🧪 Test — Classification",
    "🧪 Test — YOLO Detection",
])

# ── Home ───────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("Brain Tumor Diagnostic System")
    st.markdown("""
    This dashboard presents the training results of two models:
    - **InceptionV3** — tumor classification (4 classes)
    - **YOLOv8n** — tumor detection with bounding boxes

    Use the sidebar to navigate between results.
    """)

# ── Classification ─────────────────────────────────────────────────────────────
elif page == "🧠 Classification":
    st.title("InceptionV3 — Classification Results")

    fe = json.loads((MODELS / "results_fe.json").read_text())
    ft = json.loads((MODELS / "results_ft.json").read_text())

    # Metrics table
    st.subheader("Test Metrics")
    metrics_df = pd.DataFrame({
        "Phase":     ["Feature Extraction", "Fine Tuning"],
        "Accuracy":  [fe["test_metrics"]["accuracy"],  ft["test_metrics"]["accuracy"]],
        "Precision": [fe["test_metrics"]["precision"], ft["test_metrics"]["precision"]],
        "Recall":    [fe["test_metrics"]["recall"],    ft["test_metrics"]["recall"]],
        "F1":        [fe["test_metrics"]["f1"],        ft["test_metrics"]["f1"]],
    }).set_index("Phase")
    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

    # Training curves
    st.subheader("Training Curves")
    fe_epochs = len(fe["history"]["accuracy"])
    ft_epochs = len(ft["history"]["accuracy"])
    fe_x = list(range(1, fe_epochs + 1))
    ft_x = list(range(fe_epochs + 1, fe_epochs + ft_epochs + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in [(ax1, "accuracy", "Accuracy"), (ax2, "loss", "Loss")]:
        ax.plot(fe_x, fe["history"][metric],          color="blue",  label="Train FE")
        ax.plot(fe_x, fe["history"][f"val_{metric}"], color="blue",  linestyle="--", label="Val FE")
        ax.plot(ft_x, ft["history"][metric],          color="red",   label="Train FT")
        ax.plot(ft_x, ft["history"][f"val_{metric}"], color="red",   linestyle="--", label="Val FT")
        ax.axvline(fe_epochs, color="gray", linestyle=":", label="FT start")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
    st.pyplot(fig)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    col1, col2 = st.columns(2)
    col1.image(str(MODELS / "confusion_matrix_fe.png"), caption="Feature Extraction")
    col2.image(str(MODELS / "confusion_matrix_ft.png"), caption="Fine Tuning")

# ── YOLO ───────────────────────────────────────────────────────────────────────
elif page == "🔍 YOLO Detection":
    st.title("YOLOv8n — Detection Results")

    df = pd.read_csv(YOLO_DIR / "results.csv")
    df.columns = df.columns.str.strip()

    # Key metrics over epochs
    st.subheader("Training Curves")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plots = [
        ("metrics/mAP50(B)",      "mAP@50",   axes[0][0]),
        ("metrics/mAP50-95(B)",   "mAP@50-95",axes[0][1]),
        ("metrics/precision(B)",  "Precision", axes[1][0]),
        ("metrics/recall(B)",     "Recall",    axes[1][1]),
    ]
    for col, title, ax in plots:
        ax.plot(df["epoch"], df[col])
        ax.set_title(title); ax.set_xlabel("Epoch")
    plt.tight_layout()
    st.pyplot(fig)

    # Losses
    st.subheader("Losses")
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(axes2,
        ["train/box_loss", "train/cls_loss", "train/dfl_loss"],
        ["Box Loss", "Class Loss", "DFL Loss"]):
        ax.plot(df["epoch"], df[col],          label="Train")
        ax.plot(df["epoch"], df[col.replace("train", "val")], label="Val", linestyle="--")
        ax.set_title(title); ax.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    # Result images
    st.subheader("Result Plots")
    images = {
        "Confusion Matrix":            "confusion_matrix.png",
        "Confusion Matrix Normalized": "confusion_matrix_normalized.png",
        "PR Curve":                    "BoxPR_curve.png",
        "F1 Curve":                    "BoxF1_curve.png",
        "Precision Curve":             "BoxP_curve.png",
        "Recall Curve":                "BoxR_curve.png",
    }
    cols = st.columns(2)
    for i, (caption, fname) in enumerate(images.items()):
        cols[i % 2].image(str(YOLO_DIR / fname), caption=caption, use_container_width=True)

    st.subheader("Validation Predictions")
    val_cols = st.columns(2)
    for i in range(3):
        val_cols[0].image(str(YOLO_DIR / f"val_batch{i}_labels.jpg"), caption=f"Batch {i} — Labels")
        val_cols[1].image(str(YOLO_DIR / f"val_batch{i}_pred.jpg"),   caption=f"Batch {i} — Predictions")

# ── Test Classification ────────────────────────────────────────────────────────
elif page == "🧪 Test — Classification":
    st.title("Test — Tumor Classification")
    st.caption("Upload a brain MRI image and the InceptionV3 model will predict the tumor type.")

    uploaded = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.image(uploaded, caption="Uploaded image", width=300)
        if st.button("Classify"):
            with st.spinner("Running inference..."):
                try:
                    resp = requests.post(f"{API_CLS}/predict",
                                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)})
                    resp.raise_for_status()
                    data = resp.json()
                    st.success(f"**Prediction: {data['class'].upper()}** — confidence {data['confidence']:.2%}")
                    st.subheader("All class probabilities")
                    prob_df = pd.DataFrame(data["all_probs"].items(), columns=["Class", "Probability"])
                    st.bar_chart(prob_df.set_index("Class"))
                except Exception as e:
                    st.error(f"API error: {e}")

# ── Test YOLO ──────────────────────────────────────────────────────────────────
elif page == "🧪 Test — YOLO Detection":
    st.title("Test — Tumor Detection (YOLOv8)")
    st.caption("Upload a brain MRI image and YOLOv8 will draw bounding boxes around detected tumors.")

    uploaded = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.image(uploaded, caption="Uploaded image", width=300)
        if st.button("Detect"):
            with st.spinner("Running inference..."):
                try:
                    resp = requests.post(f"{API_DET}/predict",
                                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)})
                    resp.raise_for_status()
                    data = resp.json()
                    img_bytes = base64.b64decode(data["image_b64"])
                    st.image(Image.open(io.BytesIO(img_bytes)), caption="Detection result", use_container_width=True)
                    if data["detections"]:
                        st.subheader("Detections")
                        st.dataframe(pd.DataFrame(data["detections"]), use_container_width=True)
                    else:
                        st.info("No tumor detected.")
                except Exception as e:
                    st.error(f"API error: {e}")
