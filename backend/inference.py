"""Low-memory genuine inference for the trained PlantVillage MobileNetV2 model."""
from __future__ import annotations

import gc
import hashlib
import io
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tflite_runtime.interpreter import Interpreter

IMG_SIZE = (224, 224)
MAX_VISUAL_SIZE = (512, 512)
_INFERENCE_LOCK = __import__("threading").Lock()
logger = logging.getLogger("cropguard.inference")

MODEL_VERSION = "cropguard-plantvillage-mobilenetv2-v1"
TRAINING_SCRIPT_COMMIT = "7a570daddc44f27525ec9030756035e9129b6d29"
EXPECTED_ARCHITECTURE_HASH = "9179243efcc7202932d5275aa0a123c9c1b3d5dbb9cef7942da97d2878ec3aef"


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0


class InferenceEngine:
    """Run the genuine trained checkpoint through its TFLite deployment form.

    The Keras training graph already contains MobileNetV2's preprocess_input layer.
    Therefore inference MUST pass decoded RGB pixels in the original 0..255 range.
    Applying /127.5-1 here would preprocess the image twice and can collapse
    predictions toward one class (the reported all-Tomato failure).
    """

    def __init__(self, model_path="models/plantvillage_best.tflite", labels_path="models/labels.json"):
        model_path = Path(model_path)
        labels_path = Path(labels_path)
        if not model_path.exists():
            raise FileNotFoundError(f"LOW-MEMORY MODEL MISSING: {model_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"LABEL MAP MISSING: {labels_path}")

        self.labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if len(self.labels) != 38:
            raise RuntimeError(f"Expected 38 PlantVillage labels, found {len(self.labels)}.")

        self.interpreter = Interpreter(model_path=str(model_path), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if len(self.input_details) != 1 or len(self.output_details) != 1:
            raise RuntimeError("Unexpected TFLite model input/output structure.")

        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]
        shape = tuple(int(v) for v in self.input_details[0]["shape"])
        if shape != (1, 224, 224, 3):
            raise RuntimeError(f"Unexpected TFLite input shape: {shape}")
        if self.output_details[0]["shape"][-1] != len(self.labels):
            raise RuntimeError("TFLite output count does not match labels.json.")
        if self.input_details[0]["dtype"] is not np.float32:
            raise RuntimeError(f"Unexpected TFLite input dtype: {self.input_details[0]['dtype']}")

        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        with _INFERENCE_LOCK:
            self.interpreter.set_tensor(self.input_index, dummy)
            self.interpreter.invoke()
            warm = self.interpreter.get_tensor(self.output_index)
        if warm.shape[-1] != len(self.labels):
            raise RuntimeError("Warm-up output shape mismatch.")
        del dummy, warm
        gc.collect()
        self.gradcam_available = False
        logger.info("LOW-MEMORY TFLite model loaded; file=%s size=%d bytes current_rss=%.1fMB", model_path, model_path.stat().st_size, _rss_mb())

    @staticmethod
    def _preprocess(raw: bytes):
        try:
            with Image.open(io.BytesIO(raw)) as decoded:
                decoded.verify()
            with Image.open(io.BytesIO(raw)) as decoded:
                original = ImageOps.exif_transpose(decoded).convert("RGB")
        except Exception as exc:
            raise ValueError("Upload is not a valid decodable image.") from exc

        original.thumbnail(MAX_VISUAL_SIZE, Image.Resampling.LANCZOS)
        resized = original.resize(IMG_SIZE, Image.Resampling.BILINEAR)
        # IMPORTANT: the trained Keras graph contains MobileNetV2 preprocess_input.
        # Keep this tensor in 0..255 so the embedded preprocessing runs exactly once.
        tensor = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
        del resized
        return original, tensor

    def predict(self, raw: bytes) -> dict:
        with _INFERENCE_LOCK:
            original, tensor = self._preprocess(raw)
            try:
                self.interpreter.set_tensor(self.input_index, tensor)
                self.interpreter.invoke()
                vector = np.asarray(self.interpreter.get_tensor(self.output_index)[0], dtype=np.float32).copy()
            finally:
                del tensor
                gc.collect()

            if np.any(vector < 0.0) or not np.isfinite(vector).all():
                raise RuntimeError("Model returned invalid prediction values.")
            total = float(vector.sum())
            if total <= 0.0:
                raise RuntimeError("Model returned an empty prediction distribution.")
            probabilities = vector / total

            idx = int(np.argmax(probabilities))
            top_indices = np.argsort(probabilities)[::-1][:3]
            crop, disease = self._split_label(self.labels[idx])
            result = {
                "label": self.labels[idx],
                "crop": crop,
                "disease": disease,
                "class_index": idx,
                "confidence": float(probabilities[idx]),
                "top_predictions": [
                    {"label": self.labels[int(i)], "probability": float(probabilities[int(i)])}
                    for i in top_indices
                ],
                "image_sha256": hashlib.sha256(raw).hexdigest(),
                "gradcam_available": False,
                "gradcam_error": "Grad-CAM is disabled in low-memory deployment mode; prediction remains genuine.",
            }
            del vector, probabilities, top_indices
            gc.collect()
            return result

    @staticmethod
    def _split_label(label: str):
        parts = label.split("___", 1)
        return (parts[0], parts[1].replace("_", " ")) if len(parts) == 2 else (label.split("_", 1)[0], label)
