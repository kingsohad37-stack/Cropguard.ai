import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREATMENTS = json.loads((ROOT / "backend" / "treatments.json").read_text(encoding="utf-8"))


def normalize_treatment_key(value: str) -> str:
    value = value.strip().replace("___", "_")
    value = re.sub(r"_+", "_", value)
    value = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return value.strip("_")


def treatment_for(label: str):
    wanted = normalize_treatment_key(label)
    for key, advisory in TREATMENTS.items():
        if normalize_treatment_key(key) == wanted:
            return key, advisory
    return None, None


def test_plantvillage_treatment_key_mapping():
    expected = {
        "Tomato___Early_blight": "Tomato_Early_blight",
        "Tomato___Late_blight": "Tomato_Late_blight",
        "Potato___Early_blight": "Potato_Early_blight",
        "Potato___Late_blight": "Potato_Late_blight",
    }
    for label, treatment_key in expected.items():
        matched_key, advisory = treatment_for(label)
        assert matched_key == treatment_key
        assert isinstance(advisory, dict)
        assert advisory.get("summary")
        assert isinstance(advisory.get("actions"), list)


def test_unrelated_disease_does_not_cross_match():
    matched_key, advisory = treatment_for("Pepper___Late_blight")
    assert matched_key is None
    assert advisory is None
