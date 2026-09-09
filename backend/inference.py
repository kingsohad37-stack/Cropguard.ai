"""Real inference and Grad-CAM for the trained PlantVillage MobileNetV2 model."""
from __future__ import annotations
import gc
import io
import json
import hashlib
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

IMG_SIZE = (224, 224)
# Uploaded phone photos can be 10+ megapixels. Keeping the original image at
# full resolution makes the Grad-CAM overlay allocate huge NumPy arrays on
# Render's small instance. Prediction still uses the exact 224x224 model input.
MAX_VISUAL_SIZE = (1280, 1280)

_INFERENCE_LOCK = threading.Lock()


class InferenceEngine:
    def __init__(self, model_path="models/plantvillage_best.h5", labels_path="models/labels.json"):
        model_path = Path(model_path)
        labels_path = Path(labels_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"REAL MODEL MISSING: {model_path}. Run the real data pipeline and training first."
            )
        if not labels_path.exists():
            raise FileNotFoundError(f"LABEL MAP MISSING: {labels_path}")

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.labels = json.loads(labels_path.read_text())
        if self.model.output_shape[-1] != len(self.labels):
            raise RuntimeError(
                f"Model has {self.model.output_shape[-1]} outputs but labels.json has {len(self.labels)} labels."
            )
        self.grad_layer = self._find_last_conv_like_layer()
        input_tensor = tf.keras.Input(shape=(*IMG_SIZE, 3), name="gradcam_input")
        features = self.grad_layer(input_tensor, training=False)
        output = features
        passed_backbone = False
        for layer in self.model.layers:
            if layer is self.grad_layer:
                passed_backbone = True
                continue
            if passed_backbone:
                output = layer(output, training=False)
        self.grad_model = tf.keras.Model(input_tensor, [features, output], name="cropguard_gradcam")

    def _find_last_conv_like_layer(self):
        for layer in reversed(self.model.layers):
            try:
                shape = layer.output.shape
                if len(shape) == 4:
                    return layer
            except (AttributeError, TypeError):
                continue
        raise RuntimeError("No 4-D convolutional feature map exists for Grad-CAM.")

    @staticmethod
    def _preprocess(raw: bytes):
        try:
            with Image.open(io.BytesIO(raw)) as decoded:
                decoded.verify()
            with Image.open(io.BytesIO(raw)) as decoded:
                original = ImageOps.exif_transpose(decoded).convert("RGB")
        except Exception as exc:
            raise ValueError("Upload is not a valid decodable image.") from exc

        # Bound every image-sized allocation used by leaf masking and Grad-CAM
        # rendering while preserving the uploaded image itself for its hash.
        original.thumbnail(MAX_VISUAL_SIZE, Image.Resampling.LANCZOS)
        resized = original.resize(IMG_SIZE, Image.Resampling.BILINEAR)
        x = np.asarray(resized, dtype=np.float32)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        return original, tf.convert_to_tensor(x[None, ...], dtype=tf.float32)

    @staticmethod
    def _leaf_mask(image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image).astype(np.float32) / 255.0
        mx = rgb.max(axis=2)
        mn = rgb.min(axis=2)
        sat = (mx - mn) / (mx + 1e-6)
        green = (rgb[..., 1] > rgb[..., 0] * 0.72) & (rgb[..., 1] > rgb[..., 2] * 0.72)
        nonwhite = mx < 0.97
        mask = (green | (sat > 0.18)) & nonwhite
        if mask.mean() < 0.01:
            mask = nonwhite
        if mask.mean() < 0.01:
            mask = np.ones(mask.shape, dtype=bool)
        return mask

    def predict(self, raw: bytes) -> dict:
        with _INFERENCE_LOCK:
            try:
                original, x = self._preprocess(raw)

                with tf.GradientTape() as tape:
                    conv_features, predictions = self.grad_model(x, training=False)
                    class_index = tf.argmax(predictions[0], axis=-1)
                    class_score = predictions[:, class_index]

                gradients = tape.gradient(class_score, conv_features)
                if gradients is None:
                    raise RuntimeError("Gradient computation returned None; cannot create Grad-CAM.")

                weights = tf.reduce_mean(gradients, axis=(1, 2))
                cam = tf.reduce_sum(conv_features * weights[:, None, None, :], axis=-1)[0]
                cam = tf.maximum(cam, 0.0)
                cam = cam / (tf.reduce_max(cam) + tf.keras.backend.epsilon())
                cam_np = cam.numpy().astype(np.float32)

                heat = Image.fromarray(np.uint8(cam_np * 255), mode="L").resize(
                    original.size, Image.Resampling.BILINEAR
                )
                heat_np = np.asarray(heat, dtype=np.float32) / 255.0
                leaf_mask = self._leaf_mask(original)
                leaf_values = heat_np[leaf_mask]
                severity = float(np.mean(leaf_values) * 100.0)
                coverage = float(np.mean(leaf_values >= 0.50) * 100.0)

                idx = int(class_index.numpy())
                label = self.labels[idx]
                crop, disease = self._split_label(label)
                vector = predictions[0].numpy()
                top_indices = np.argsort(vector)[::-1][: min(3, len(self.labels))]

                heatmap_png = self._overlay(original, cam_np)
                return {
                    "label": label,
                    "crop": crop,
                    "disease": disease,
                    "class_index": idx,
                    "confidence": float(predictions[0, idx].numpy()),
                    "top_predictions": [
                        {"label": self.labels[int(i)], "probability": float(vector[int(i)])}
                        for i in top_indices
                    ],
                    "image_sha256": hashlib.sha256(raw).hexdigest(),
                    "severity_score": severity,
                    "heatmap_coverage_percent": coverage,
                    "heatmap_png": heatmap_png,
                }
            finally:
                gc.collect()

    @staticmethod
    def _split_label(label: str):
        parts = label.split("___", 1)
        if len(parts) == 2:
            return parts[0], parts[1].replace("_", " ")
        return label.split("_", 1)[0], label

    @staticmethod
    def _overlay(image: Image.Image, cam: np.ndarray) -> bytes:
        heat_small = Image.fromarray(np.uint8(cam * 255), mode="L")
        heat = heat_small.resize(image.size, Image.Resampling.BILINEAR)
        arr = np.asarray(heat, dtype=np.float32) / 255.0

        r = np.clip(1.5 - np.abs(4.0 * arr - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * arr - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * arr - 1.0), 0.0, 1.0)
        heat_rgb = np.stack([r, g, b], axis=-1)
        base = np.asarray(image, dtype=np.float32) / 255.0
        overlay = np.clip(base * 0.58 + heat_rgb * 0.42, 0.0, 1.0)
        out = Image.fromarray(np.uint8(overlay * 255.0), mode="RGB")

        buf = io.BytesIO()
        out.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
