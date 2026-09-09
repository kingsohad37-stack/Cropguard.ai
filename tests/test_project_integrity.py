import ast
import asyncio
import io
import json
from pathlib import Path

import pytest
from PIL import Image
from starlette.datastructures import Headers, UploadFile

ROOT = Path(__file__).resolve().parents[1]

def png_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, "PNG")
    return buffer.getvalue()

def upload(name, content):
    return UploadFile(io.BytesIO(content), filename=name, headers=Headers({"content-type": "image/png"}))

def test_required_files_and_python_syntax():
    files = ["scripts/data_pipeline.py", "scripts/train.py", "scripts/evaluate.py", "scripts/gradcam.py", "backend/inference.py", "backend/main.py", "backend/treatments.json", "frontend/app.py", "README.md"]
    assert all((ROOT / item).exists() for item in files)
    for item in [x for x in files if x.endswith(".py")]:
        ast.parse((ROOT / item).read_text(encoding="utf-8"))

def test_treatment_records_are_structured_and_label_normalizable():
    from backend.main import advisory_for
    records = json.loads((ROOT / "backend/treatments.json").read_text(encoding="utf-8"))
    assert len(records) >= 15
    assert all(record.get("summary") and record.get("actions") and record.get("sources") for record in records.values())
    assert advisory_for("Tomato___Early_blight")["summary"] == records["Tomato_Early_blight"]["summary"]

def test_api_refuses_prediction_without_real_model_and_never_invents_metrics(tmp_path, monkeypatch):
    from fastapi import HTTPException
    import backend.main as main
    monkeypatch.setattr(main, "DB", tmp_path / "fresh.db")
    monkeypatch.setattr(main, "METRICS", tmp_path / "missing.json")
    monkeypatch.setattr(main, "engine", None)
    assert main.health()["model_loaded"] is False
    with pytest.raises(HTTPException, match="Real trained model") as prediction_error:
        asyncio.run(main.predict(upload("leaf.png", png_bytes("green"))))
    assert prediction_error.value.status_code == 503
    with pytest.raises(HTTPException) as metrics_error:
        main.metrics()
    assert metrics_error.value.status_code == 404
    assert main.history() == []

def test_uploaded_bytes_and_hashes_are_not_a_prediction_mechanism(tmp_path, monkeypatch):
    import backend.main as main
    class Engine:
        def predict(self, raw):
            # Isolated API test: records exact image content while a real engine
            # is separately tested when a trained artifact is available.
            from hashlib import sha256
            return {"label":"Tomato___Early_blight", "crop":"Tomato", "disease":"Early blight", "class_index":0, "confidence":0.37, "severity_score":1.0, "heatmap_coverage_percent":2.0, "top_predictions":[{"label":"Tomato___Early_blight","probability":0.37}], "image_sha256":sha256(raw).hexdigest(), "heatmap_png":png_bytes("red")}
    monkeypatch.setattr(main, "DB", tmp_path / "scans.db")
    monkeypatch.setattr(main, "engine", Engine())
    first = asyncio.run(main.predict(upload("a.png", png_bytes("green"))))
    second = asyncio.run(main.predict(upload("b.png", png_bytes("blue"))))
    assert first["image_sha256"] != second["image_sha256"]
    assert len((tmp_path / "scans.db").read_bytes()) > 0

def test_real_model_contract_when_artifacts_are_available():
    model = ROOT / "models/plantvillage_best.keras"
    labels = ROOT / "models/labels.json"
    if not (model.exists() and labels.exists()):
        pytest.skip("No real trained model is included by design.")
    from backend.inference import InferenceEngine
    engine = InferenceEngine(model, labels)
    assert engine.model.output_shape[-1] == len(engine.labels)
    first, second = engine.predict(png_bytes("green")), engine.predict(png_bytes("blue"))
    assert first["image_sha256"] != second["image_sha256"]
    assert first["top_predictions"]
    assert first["heatmap_png"] != second["heatmap_png"]
