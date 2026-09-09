from __future__ import annotations
import base64, json, re, time
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
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
store = ScanStore(DB)


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
    # Tests and local deployments may redirect the SQLite path after module
    # import; retain the cloud client when configured, otherwise follow DB.
    if not store.remote and store.sqlite_path != DB:
        store = ScanStore(DB)
    store.initialize()


app = FastAPI(title="CropGuard AI", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    global engine, model_error
    init_db()
    try:
        engine = InferenceEngine(str(ROOT / "models" / "plantvillage_best.keras"), str(ROOT / "models" / "labels.json"))
        model_error = None
    except Exception as exc:
        engine = None
        model_error = str(exc)
        # Health remains available so the UI can explain exactly what is missing.


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": engine is not None, "model_error": model_error}


@app.get("/metrics")
def metrics():
    if not METRICS.exists():
        raise HTTPException(404, "No real training metrics found. Run scripts/train.py first.")
    return json.loads(METRICS.read_text(encoding="utf-8"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if engine is None:
        raise HTTPException(503, "Real trained model is not loaded. Train the PlantVillage model first.")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty image")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is too large (maximum upload size is 15 MB).")
    try:
        result = engine.predict(raw)
    except Exception as exc:
        raise HTTPException(422, f"Inference failed: {exc}") from exc

    result["advisory"] = advisory_for(result["label"])
    # Keep persistence reliable if an ASGI host invokes this endpoint before
    # startup hooks have run (for example, a worker restart or direct test).
    init_db()
    try:
        result["scan_id"] = store.insert({**result, "ts": time.time(), "filename": file.filename or "upload"})
    except Exception as exc:
        raise HTTPException(503, f"Scan storage failed: {exc}") from exc

    result["heatmap_data_url"] = "data:image/png;base64," + base64.b64encode(result.pop("heatmap_png")).decode()
    return result


@app.get("/history")
def history():
    init_db()
    try:
        return store.history()
    except Exception as exc:
        raise HTTPException(503, f"Scan storage failed: {exc}") from exc
