from __future__ import annotations

import json
import zipfile
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
KERAS_MODEL = ROOT / "models" / "plantvillage_best.keras"
TFLITE_MODEL = ROOT / "models" / "plantvillage_best.tflite"
META_OUT = ROOT / "models" / "tflite_metadata.json"
LABELS = ROOT / "models" / "labels.json"

if not KERAS_MODEL.exists():
    raise FileNotFoundError(KERAS_MODEL)

with zipfile.ZipFile(KERAS_MODEL, "r") as z:
    source_metadata = json.loads(z.read("metadata.json"))

model = tf.keras.models.load_model(KERAS_MODEL, compile=False)
if model.output_shape[-1] != len(json.loads(LABELS.read_text())):
    raise RuntimeError("Keras checkpoint output count does not match labels.json")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Float32 TFLite preserves the trained weights and avoids introducing
# post-training quantization changes to the prediction behavior.
tflite_bytes = converter.convert()
TFLITE_MODEL.write_bytes(tflite_bytes)

interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL), num_threads=1)
interpreter.allocate_tensors()
inputs = interpreter.get_input_details()
outputs = interpreter.get_output_details()
if len(inputs) != 1 or len(outputs) != 1:
    raise RuntimeError(f"Unexpected TFLite IO: inputs={len(inputs)} outputs={len(outputs)}")

meta = {
    "model_version": source_metadata.get("model_version"),
    "training_script_commit": source_metadata.get("training_script_commit"),
    "architecture_hash": source_metadata.get("architecture_hash"),
    "source_checkpoint": "plantvillage_best.keras",
    "format": "tflite-float32",
    "input_shape": inputs[0]["shape"].tolist(),
    "input_dtype": str(inputs[0]["dtype"]),
    "output_shape": outputs[0]["shape"].tolist(),
    "output_dtype": str(outputs[0]["dtype"]),
    "file_size_bytes": len(tflite_bytes),
}
META_OUT.write_text(json.dumps(meta, indent=2) + "\n")
print(f"Created {TFLITE_MODEL} ({len(tflite_bytes)} bytes)")
print(json.dumps(meta, indent=2))
