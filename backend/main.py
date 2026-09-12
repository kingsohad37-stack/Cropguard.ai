from __future__ import annotations
import base64, json, re, time, logging, traceback
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.inference import InferenceEngine
from backend.storage import ScanStore

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "backend" / "scans.db"
TREAT = ROOT / "backend" / "treatments.json"
METRICS = ROOT / "models" / "metrics.json"
TREATMENTS = json.loads(TREAT.read_text(encoding="utf-8")) if TREAT.exists() else {}
engine: InferenceEngine | None = None
model_error: str | None = None
model_traceback: str | None = None
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
store = ScanStore(DB)
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


def init_db():
    global store
    if not store.remote and store.sqlite_path != DB:
        store = ScanStore(DB)
    store.initialize()


app = FastAPI(title="CropGuard AI", version="1.1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    global engine, model_error, model_traceback
    init_db()
    try:
        # InferenceEngine raises only for failures that prevent genuine model
        # prediction. Grad-CAM construction is intentionally non-fatal inside it.
        engine = InferenceEngine(str(ROOT / "models" / "plantvillage_best.keras"), str(ROOT / "models" / "labels.json"))
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
    """Run genuine TensorFlow inference without making Grad-CAM or scan persistence a hard dependency."""
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

    try:
        init_db()
        result["scan_id"] = store.insert({**result, "ts": time.time(), "filename": file.filename or "upload"})
        result["storage_status"] = "saved"
    except Exception as exc:
        logger.exception("Optional scan storage failed; returning valid inference result")
        result["scan_id"] = None
        result["storage_status"] = "unavailable"
        result["storage_error"] = str(exc)

    heatmap_png = result.pop("heatmap_png", None)
    if heatmap_png is not None:
        result["heatmap_data_url"] = "data:image/png;base64," + base64.b64encode(heatmap_png).decode()
    else:
        result["heatmap_data_url"] = None
        result["heatmap_status"] = "Grad-CAM unavailable; prediction returned without heatmap."

    logger.info(
        "Prediction complete: label=%s confidence=%.4f elapsed=%.2fs storage=%s gradcam=%s",
        result["label"], result["confidence"], time.perf_counter() - started,
        result["storage_status"], result.get("gradcam_available", False),
    )
    return result


@app.get("/history")
def history():
    try:
        init_db()
        return store.history()
    except Exception as exc:
        logger.exception("Scan history unavailable")
        raise HTTPException(503, f"Scan history is unavailable: {exc}") from exc
