from __future__ import annotations

import base64
import json
import logging
import os
import re
import resource
import time
import traceback
from pathlib import Path
from urllib.request import Request, urlopen

from fastapi import FastAPI, File, HTTPException, Request as FastAPIRequest, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
MAX_REQUEST_SECONDS = 55.0
logger = logging.getLogger("cropguard")


def _memory_mb() -> float:
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return -1.0


def _ensure_model():
    if MODEL.exists() and MODEL.stat().st_size > 1_000_000:
        logger.info("Using committed REAL trained model: %s (%d bytes); rss_max=%.1fMB", MODEL, MODEL.stat().st_size, _memory_mb())
        return
    url = os.getenv("CROPGUARD_MODEL_URL", "").strip()
    if not url:
        raise FileNotFoundError(f"REAL MODEL MISSING: {MODEL}. No valid checkpoint is present in the deployment image and CROPGUARD_MODEL_URL is not configured.")
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL.with_suffix(".keras.download")
    logger.info("Committed model missing; downloading existing trained CropGuard model as fallback")
    req = Request(url, headers={"User-Agent": "CropGuard/1.0"})
    with urlopen(req, timeout=180) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if tmp.stat().st_size <= 1_000_000:
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
    return {"summary": "No class-specific advisory is available in the reference database.", "actions": ["Inspect additional leaves and the surrounding crop area.", "Use integrated pest/disease management and avoid unnecessary pesticide applications.", "Follow only locally registered product labels and agricultural-extension recommendations."], "sources": []}


app = FastAPI(title="CropGuard AI", version="1.1.3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def safety_middleware(request: FastAPIRequest, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
        logger.info("HTTP %s %s -> %s elapsed=%.3fs rss_max=%.1fMB", request.method, request.url.path, response.status_code, time.perf_counter() - started, _memory_mb())
        return response
    except Exception as exc:
        logger.exception("Unhandled API exception: %s %s elapsed=%.3fs rss_max=%.1fMB", request.method, request.url.path, time.perf_counter() - started, _memory_mb())
        return JSONResponse(status_code=500, content={"error": "CropGuard backend failed safely instead of crashing.", "detail": f"{type(exc).__name__}: {exc}", "path": request.url.path})


@app.on_event("startup")
def startup():
    global engine, model_error, model_traceback
    startup_started = time.perf_counter()
    logger.info("CropGuard API starting; rss_max=%.1fMB model=%s", _memory_mb(), MODEL)
    try:
        _ensure_model()
        engine = InferenceEngine(str(MODEL), str(ROOT / "models" / "labels.json"))
        model_error = None
        model_traceback = None
        logger.info("Real prediction model loaded and warmed successfully; startup=%.2fs rss_max=%.1fMB gradcam=%s", time.perf_counter() - startup_started, _memory_mb(), bool(engine.grad_model))
    except Exception as exc:
        engine = None
        model_error = f"{type(exc).__name__}: {exc}"
        model_traceback = traceback.format_exc()
        logger.exception("Real prediction model failed to load; rss_max=%.1fMB", _memory_mb())


@app.get("/health")
def health():
    return {"status": "ok" if engine is not None else "degraded", "model_loaded": engine is not None, "gradcam_available": bool(engine is not None and engine.grad_model is not None), "model_error": model_error, "memory_rss_max_mb": round(_memory_mb(), 1)}


@app.get("/healthz")
def healthz():
    return {"status": "alive", "memory_rss_max_mb": round(_memory_mb(), 1)}


@app.get("/metrics")
def metrics():
    if not METRICS.exists():
        raise HTTPException(404, "No real training metrics found. Run scripts/train.py first.")
    return json.loads(METRICS.read_text(encoding="utf-8"))


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
        raise HTTPException(413, "Image is too large (maximum upload size is 15 MB).")
    logger.info("Starting real prediction: filename=%s bytes=%d rss_max=%.1fMB", file.filename or "upload", len(raw), _memory_mb())
    try:
        result = engine.predict(raw)
    except MemoryError as exc:
        logger.exception("Inference hit Python memory exhaustion; rss_max=%.1fMB", _memory_mb())
        raise HTTPException(503, "Inference ran out of memory. The model is still loaded; please retry with a smaller image.") from exc
    except Exception as exc:
        logger.exception("Real inference failed; rss_max=%.1fMB", _memory_mb())
        raise HTTPException(422, f"Inference failed safely: {type(exc).__name__}: {exc}") from exc
    elapsed = time.perf_counter() - started
    if elapsed >= MAX_REQUEST_SECONDS:
        logger.warning("Prediction exceeded safety budget: %.2fs", elapsed)
    result["advisory"] = advisory_for(result["label"])
    heatmap_png = result.pop("heatmap_png", None)
    if heatmap_png is not None:
        result["heatmap_data_url"] = "data:image/png;base64," + base64.b64encode(heatmap_png).decode()
    else:
        result["heatmap_data_url"] = None
        result["heatmap_status"] = "Grad-CAM unavailable; prediction returned without heatmap."
    logger.info("Prediction complete: label=%s confidence=%.4f elapsed=%.2fs gradcam=%s rss_max=%.1fMB", result["label"], result["confidence"], elapsed, result.get("gradcam_available", False), _memory_mb())
    return result
