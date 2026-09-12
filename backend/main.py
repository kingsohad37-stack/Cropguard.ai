from __future__ import annotations

import base64
import json
import logging
import re
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import InferenceEngine

ROOT = Path(__file__).resolve().parents[1]
TREAT = ROOT / "backend" / "treatments.json"
METRICS = ROOT / "models" / "metrics.json"
TREATMENTS = json.loads(TREAT.read_text(encoding="utf-8")) if TREAT.exists() else {}
engine: InferenceEngine | None = None
model_error: str | None = None
model_traceback: str | None = None
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
logger = logging.getLogger("cropguard")


def advisory_for(label: str):
    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    wanted = normalize(label)
    for key, advisory in TREATMENTS.items():
        if normalize(key) == wanted:
            return advisory
    return {
        "summary": "No class-specific advisory is available in the reference database.",
        "actions": [
            "Inspect additional leaves and the surrounding crop area.",
            "Use integrated pest/disease management and avoid unnecessary pesticide applications.",
            "Follow only locally registered product labels and agricultural-extension recommendations.",
        ],
        "sources": [],
    }


app = FastAPI(title="CropGuard AI", version="1.1.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    global engine, model_error, model_traceback
    try:
        engine = InferenceEngine(
            str(ROOT / "models" / "plantvillage_best.keras"),
            str(ROOT / "models" / "labels.json"),
        )
        model_error = None
        model_traceback = None
        logger.info("Real prediction model loaded and warmed successfully")
    except Exception as exc:
        engine = None
        model_error = f"{type(exc).__name__}: {exc}"
        model_traceback = traceback.format_exc()
        logger.exception("Real prediction model failed to load")


@app.get("/health")
def health():
    gradcam_available = bool(engine is not None and engine.grad_model is not None)
    return {
        "status": "ok" if engine is not None else "degraded",
        "model_loaded": engine is not None,
        "gradcam_available": gradcam_available,
        "model_error": model_error,
        "model_traceback": model_traceback,
    }


@app.get("/metrics")
def metrics():
    if not METRICS.exists():
        raise HTTPException(404, "No real training metrics found. Run scripts/train.py first.")
    return json.loads(METRICS.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Run genuine TensorFlow inference without storing the uploaded scan or result."""
    started = time.perf_counter()
    if engine is None:
        raise HTTPException(503, "Real trained model is not loaded. Check /health for the model error.")

    raw = file.file.read()
    if not raw:
        raise HTTPException(400, "Empty image")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is too large (maximum upload size is 15 MB).")

    logger.info("Starting real prediction: filename=%s bytes=%d", file.filename or "upload", len(raw))
    try:
        result = engine.predict(raw)
    except Exception as exc:
        logger.exception("Real inference failed")
        raise HTTPException(422, f"Inference failed: {exc}") from exc

    result["advisory"] = advisory_for(result["label"])

    heatmap_png = result.pop("heatmap_png", None)
    if heatmap_png is not None:
        result["heatmap_data_url"] = "data:image/png;base64," + base64.b64encode(heatmap_png).decode()
    else:
        result["heatmap_data_url"] = None
        result["heatmap_status"] = "Grad-CAM unavailable; prediction returned without heatmap."

    logger.info(
        "Prediction complete: label=%s confidence=%.4f elapsed=%.2fs gradcam=%s",
        result["label"],
        result["confidence"],
        time.perf_counter() - started,
        result.get("gradcam_available", False),
    )
    return result
