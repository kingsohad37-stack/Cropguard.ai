from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import traceback
from pathlib import Path
from urllib.request import Request, urlopen

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import InferenceEngine

ROOT = Path(__file__).resolve().parents[1]
TREAT = ROOT / "backend" / "treatments.json"
METRICS = ROOT / "models" / "metrics.json"
MODEL = ROOT / "models" / "plantvillage_best.keras"
TREATMENTS = json.loads(TREAT.read_text(encoding="utf-8")) if TREAT.exists() else {}
engine: InferenceEngine | None = None
model_error: str | None = None
model_traceback: str | None = None
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
logger = logging.getLogger("cropguard")


def _ensure_model():
    """Fetch the already-trained model into the deployment image when it is absent."""
    if MODEL.exists() and MODEL.stat().st_size > 20_000_000:
        return
    url = os.getenv("CROPGUARD_MODEL_URL", "").strip()
    if not url:
        raise FileNotFoundError(f"REAL MODEL MISSING: {MODEL}. CROPGUARD_MODEL_URL is not configured.")
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL.with_suffix(".keras.download")
    logger.info("Downloading existing trained CropGuard model for deployment")
    req = Request(url, headers={"User-Agent": "CropGuard/1.0"})
    with urlopen(req, timeout=180) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if tmp.stat().st_size <= 20_000_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded model is unexpectedly small or incomplete.")
    tmp.replace(MODEL)
    logger.info("Existing trained model downloaded successfully: %d bytes", MODEL.stat().st_size)


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
        _ensure_model()
        engine = InferenceEngine(str(MODEL), str(ROOT / "models" / "labels.json"))
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
