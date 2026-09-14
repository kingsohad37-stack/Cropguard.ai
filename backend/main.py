from __future__ import annotations

import base64
import json
import logging
import os
import resource
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request as FastAPIRequest, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.inference import InferenceEngine

ROOT = Path(__file__).resolve().parents[1]
TREAT = ROOT / "backend" / "treatments.json"
METRICS = ROOT / "models" / "metrics.json"
MODEL = ROOT / "models" / "plantvillage_best.tflite"
TREATMENTS = json.loads(TREAT.read_text(encoding="utf-8")) if TREAT.exists() else {}
engine: InferenceEngine | None = None
model_error: str | None = None
model_traceback: str | None = None
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_REQUEST_SECONDS = 55.0
logger = logging.getLogger("cropguard")


def _memory_mb() -> float:
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return -1.0


def _current_rss_mb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0


def _ensure_model():
    if MODEL.exists() and MODEL.stat().st_size > 1_000_000:
        logger.info("Using genuine low-memory TFLite model: %s (%d bytes)", MODEL, MODEL.stat().st_size)
        return
    raise FileNotFoundError(
        f"LOW-MEMORY MODEL MISSING: {MODEL}. The deployment must contain the generated TFLite representation of the committed trained checkpoint."
    )


def _normalize_treatment_key(value: str) -> str:
    value = value.strip().replace("___", "_")
    import re
    value = re.sub(r"_+", "_", value)
    value = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return value.strip("_")


def advisory_for(label: str) -> dict:
    wanted = _normalize_treatment_key(label)
    candidates = {wanted}
    if "___" in label:
        crop_raw, disease_raw = label.split("___", 1)
    else:
        pieces = label.split("_", 1)
        crop_raw = pieces[0]
        disease_raw = pieces[1] if len(pieces) == 2 else ""
    crop = _normalize_treatment_key(crop_raw)
    disease = _normalize_treatment_key(disease_raw)
    if crop and disease:
        if disease.startswith(f"{crop}_"):
            disease = disease[len(crop) + 1 :]
        candidates.add(f"{crop}_{disease}")
    for key, advisory in TREATMENTS.items():
        if _normalize_treatment_key(key) in candidates:
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


app = FastAPI(title="CropGuard AI", version="1.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def safety_middleware(request: FastAPIRequest, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
        logger.info(
            "HTTP %s %s -> %s elapsed=%.3fs rss_max=%.1fMB current_rss=%.1fMB",
            request.method, request.url.path, response.status_code,
            time.perf_counter() - started, _memory_mb(), _current_rss_mb(),
        )
        return response
    except Exception as exc:
        logger.exception("Unhandled API exception: %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"error": "CropGuard backend failed safely.", "detail": f"{type(exc).__name__}: {exc}", "path": request.url.path})


@app.on_event("startup")
def startup():
    global engine, model_error, model_traceback
    startup_started = time.perf_counter()
    logger.info("CropGuard API starting with low-memory TFLite runtime")
    try:
        _ensure_model()
        engine = InferenceEngine(str(MODEL), str(ROOT / "models" / "labels.json"))
        model_error = None
        model_traceback = None
        logger.info(
            "Genuine TFLite model loaded and warmed; startup=%.2fs rss_max=%.1fMB current_rss=%.1fMB",
            time.perf_counter() - startup_started, _memory_mb(), _current_rss_mb(),
        )
    except Exception as exc:
        engine = None
        model_error = f"{type(exc).__name__}: {exc}"
        model_traceback = traceback.format_exc()
        logger.exception("Prediction model failed to load")


@app.get("/health")
def health():
    return {
        "status": "ok" if engine is not None else "degraded",
        "model_loaded": engine is not None,
        "inference_runtime": "tflite-runtime" if engine is not None else None,
        "gradcam_available": False,
        "model_error": model_error,
        "memory_rss_max_mb": round(_memory_mb(), 1),
        "memory_rss_current_mb": round(_current_rss_mb(), 1),
    }


@app.get("/healthz")
def healthz():
    return {"status": "alive", "model_loaded": engine is not None, "memory_rss_current_mb": round(_current_rss_mb(), 1)}


@app.get("/metrics")
def metrics():
    if not METRICS.exists():
        raise HTTPException(404, "No real training metrics found. Run scripts/train.py first.")
    return json.loads(METRICS.read_text(encoding="utf-8"))


@app.get("/history")
def history():
    # There is no database in this deployment, so never fabricate persistent history.
    return []


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    started = time.perf_counter()
    if engine is None:
        detail = model_error or "unknown model startup error"
        raise HTTPException(503, f"Real trained model is not ready: {detail}. Retry after /health reports model_loaded=true.")

    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(400, "Empty image")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is too large (maximum upload size is 8 MB).")

    logger.info("Starting genuine prediction: filename=%s bytes=%d current_rss=%.1fMB", file.filename or "upload", len(raw), _current_rss_mb())
    try:
        result = engine.predict(raw)
    except MemoryError as exc:
        logger.exception("Inference hit memory exhaustion; current_rss=%.1fMB", _current_rss_mb())
        raise HTTPException(503, "Inference temporarily exhausted memory. Please retry with a smaller image.") from exc
    except Exception as exc:
        logger.exception("Genuine inference failed safely; current_rss=%.1fMB", _current_rss_mb())
        raise HTTPException(422, f"Inference failed safely: {type(exc).__name__}: {exc}") from exc
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    elapsed = time.perf_counter() - started
    if elapsed >= MAX_REQUEST_SECONDS:
        logger.warning("Prediction exceeded safety budget: %.2fs", elapsed)
    result["advisory"] = advisory_for(result["label"])
    result["processing_seconds"] = round(elapsed, 3)
    return result


@app.get("/")
def root():
    return {"name": "CropGuard AI", "status": "online", "health": "/health", "predict": "/predict", "history": "/history"}
