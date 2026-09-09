"""Streamlit client for genuine CropGuard inference only."""
import os
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title="CropGuard AI", page_icon="🌿", layout="wide")
st.title("🌿 CropGuard AI")
st.caption("PlantVillage-trained MobileNetV2 • real inference • live Grad-CAM")

with st.sidebar:
    configured_api = os.environ.get("CROPGUARD_API_URL", "http://localhost:8000").strip().rstrip("/")
    if configured_api and not configured_api.startswith(("http://", "https://")):
        configured_api = "https://" + configured_api
    api = st.text_input("Backend URL", configured_api).strip().rstrip("/")
    health = None
    try:
        health_response = requests.get(f"{api}/health", timeout=20)
        health_response.raise_for_status()
        health = health_response.json()
        if health.get("model_loaded"):
            st.success("Real trained model loaded")
        else:
            st.warning("Model unavailable — train it before scanning.")
            st.caption(health.get("model_error", "No model loaded."))
    except (requests.RequestException, ValueError) as exc:
        st.error("Backend unavailable")
        st.caption(str(exc))

st.subheader("📷 Upload a leaf image")
st.caption("Drag and drop a JPG, JPEG, PNG, or WEBP image into the box below, or tap Browse files.")
upload = st.file_uploader(
    "Drop your leaf image here",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
    key="leaf_upload",
)

st.caption("Or use your camera")
camera = st.camera_input("Take a leaf photo", key="leaf_camera")

# Prefer an explicitly uploaded file. Fall back to the camera capture.
source = upload if upload is not None else camera

if source is not None:
    try:
        image_bytes = source.getvalue()
        image = Image.open(source).convert("RGB")
        st.success(f"Image received: {source.name or 'camera photo'}")
        st.image(image, caption="Your uploaded leaf image", width=500)
    except (UnidentifiedImageError, OSError):
        st.error("That file is not a valid image. Please choose a JPEG, PNG, WEBP, or camera image.")
        source = None
        image_bytes = None
else:
    image_bytes = None

if source is not None and image_bytes is not None and st.button(
    "🔬 Analyze with trained model",
    type="primary",
    disabled=not bool(health and health.get("model_loaded")),
):
    try:
        with st.spinner("Running TensorFlow inference and Grad-CAM…"):
            response = requests.post(
                f"{api}/predict",
                files={
                    "file": (
                        source.name or "camera.jpg",
                        image_bytes,
                        source.type or "image/jpeg",
                    )
                },
                timeout=180,
            )
        if response.status_code == 503:
            st.error("A real trained model is required before predictions can be made.")
        else:
            response.raise_for_status()
            data = response.json()
            c1, c2, c3 = st.columns(3)
            c1.metric("Crop", data["crop"])
            c2.metric("Diagnosis", data["disease"])
            c3.metric("Model confidence", f"{data['confidence'] * 100:.2f}%")
            st.metric("AI-derived severity estimate", f"{data['severity_score']:.2f}%")
            st.caption(f"Leaf area above activation threshold: {data['heatmap_coverage_percent']:.2f}% • Image SHA-256: {data['image_sha256']}")
            st.image(data["heatmap_data_url"], caption="Grad-CAM derived from this uploaded image")
            st.subheader("Other model predictions")
            st.table([{"class": p["label"], "probability": f"{p['probability'] * 100:.2f}%"} for p in data["top_predictions"]])
            advisory = data["advisory"]
            st.subheader("Treatment advisory")
            st.write(advisory["summary"])
            for action in advisory["actions"]:
                st.write(f"• {action}")
            for source_link in advisory.get("sources", []):
                st.caption(source_link)
    except requests.RequestException as exc:
        st.error(f"Backend/inference error: {exc}")

st.divider()
st.subheader("Model evidence")
try:
    response = requests.get(f"{api}/metrics", timeout=20)
    if response.ok:
        metrics = response.json()
        st.write(f"Measured held-out test accuracy: **{metrics['test_accuracy'] * 100:.2f}%**")
    else:
        st.info("No measured metrics yet. Run the real training and evaluation pipeline first.")
except requests.RequestException:
    st.info("Start the FastAPI backend to view measured metrics.")

st.subheader("Actual scan history")
try:
    response = requests.get(f"{api}/history", timeout=20)
    if response.ok:
        history = response.json()
        st.dataframe(history, use_container_width=True) if history else st.info("No scans yet.")
    else:
        st.info("Scan history is unavailable right now.")
except requests.RequestException:
    st.info("Start the FastAPI backend to view scan history.")

st.divider()
st.caption("CropGuard uses genuine model inference only; no prediction results or confidence scores are hardcoded.")
