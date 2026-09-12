"""Real inference and class activation maps for the trained PlantVillage MobileNetV2 model."""
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
MAX_VISUAL_SIZE = (512, 512)
_INFERENCE_LOCK = threading.Lock()


class InferenceEngine:
    def __init__(self, model_path="models/plantvillage_best.keras", labels_path="models/labels.json"):
        model_path = Path(model_path)
        labels_path = Path(labels_path)
        print(f"[CropGuard] REAL checkpoint path: {model_path} exists={model_path.exists()} size={model_path.stat().st_size if model_path.exists() else 0}", flush=True)
        if not model_path.exists():
            raise FileNotFoundError(f"REAL MODEL MISSING: {model_path}. Run the real data pipeline and training first.")
        if not labels_path.exists():
            raise FileNotFoundError(f"LABEL MAP MISSING: {labels_path}")

        print(f"[CropGuard] Loading REAL trained checkpoint: {model_path}", flush=True)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.labels = json.loads(labels_path.read_text())
        if self.model.output_shape[-1] != len(self.labels):
            raise RuntimeError(f"Model has {self.model.output_shape[-1]} outputs but labels.json has {len(self.labels)} labels.")

        self.base_model = self._find_mobilenet_base()
        self.pooling = self._find_layer(tf.keras.layers.GlobalAveragePooling2D)
        self.batch_norm = self._find_layer(tf.keras.layers.BatchNormalization)
        self.classifier = self._find_classifier()
        kernel = self.classifier.kernel
        if kernel.shape[0] != self.base_model.output_shape[-1]:
            raise RuntimeError("Classifier and MobileNetV2 feature dimensions do not match for CAM.")
        self.classifier_weights = tf.convert_to_tensor(kernel, dtype=tf.float32)
        self.classifier_bias = tf.convert_to_tensor(self.classifier.bias, dtype=tf.float32) if self.classifier.use_bias else tf.zeros((len(self.labels),), dtype=tf.float32)

        # Render has a tight memory limit. The previous implementation ran the
        # complete model AND the nested MobileNetV2 backbone for every image.
        # That duplicated the forward-pass memory and could push the free Render
        # instance to its limit, causing the HTTP connection to be terminated
        # before Streamlit received the complete response ("Response ended prematurely").
        # We now run the real trained backbone once, then execute its exact trained
        # head (GAP -> BatchNorm -> Dense) from those feature maps. This preserves
        # the trained model's predictions while using the same feature maps for CAM.
        dummy = tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)
        with _INFERENCE_LOCK, tf.device("/CPU:0"):
            preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(dummy)
            features = self.base_model(preprocessed, training=False)
            pooled = self.pooling(features)
            normalized = self.batch_norm(pooled, training=False)
            _ = self.classifier(normalized, training=False)
        del dummy, preprocessed, features, pooled, normalized
        gc.collect()
        print(f"[CropGuard] REAL TRAINED MODEL LOADED successfully: {model_path}", flush=True)

    def _find_mobilenet_base(self):
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
                return layer
        raise RuntimeError("Trained MobileNetV2 backbone was not found in the checkpoint.")

    def _find_layer(self, layer_type):
        for layer in self.model.layers:
            if isinstance(layer, layer_type):
                return layer
        raise RuntimeError(f"Trained head layer {layer_type.__name__} was not found in the checkpoint.")

    def _find_classifier(self):
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Dense) and layer.units == len(self.labels):
                return layer
        raise RuntimeError("Final trained Dense classifier was not found in the checkpoint.")

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
        x = np.asarray(resized, dtype=np.float32)
        return original, tf.convert_to_tensor(x[None, ...], dtype=tf.float32)

    def predict(self, raw: bytes) -> dict:
        with _INFERENCE_LOCK:
            try:
                original, x = self._preprocess(raw)
                with tf.device("/CPU:0"):
                    # Match the exact preprocessing used during PlantVillage
                    # training, then run the real trained backbone only once.
                    model_input = tf.keras.applications.mobilenet_v2.preprocess_input(x)
                    feature_maps = self.base_model(model_input, training=False)
                    pooled = self.pooling(feature_maps)
                    normalized = self.batch_norm(pooled, training=False)
                    predictions = self.classifier(normalized, training=False)

                class_index = tf.argmax(predictions[0], axis=-1)
                idx = int(class_index.numpy())
                weights = self.classifier_weights[:, idx]
                cam = tf.reduce_sum(feature_maps[0] * weights[None, None, :], axis=-1)
                cam = tf.maximum(cam, 0.0)
                max_value = tf.reduce_max(cam)
                cam = cam / (max_value + tf.keras.backend.epsilon())
                cam_np = cam.numpy().astype(np.float32)

                heat = Image.fromarray(np.uint8(cam_np * 255), mode="L").resize(original.size, Image.Resampling.BILINEAR)
                heat_np = np.asarray(heat, dtype=np.float32) / 255.0
                rgb = np.asarray(original).astype(np.float32) / 255.0
                mx = rgb.max(axis=2)
                mn = rgb.min(axis=2)
                sat = (mx - mn) / (mx + 1e-6)
                green = (rgb[..., 1] > rgb[..., 0] * 0.72) & (rgb[..., 1] > rgb[..., 2] * 0.72)
                leaf_mask = (green | (sat > 0.18)) & (mx < 0.97)
                if leaf_mask.mean() < 0.01:
                    leaf_mask = mx < 0.97
                if leaf_mask.mean() < 0.01:
                    leaf_mask = np.ones(leaf_mask.shape, dtype=bool)
                leaf_values = heat_np[leaf_mask]
                severity = float(np.mean(leaf_values) * 100.0)
                coverage = float(np.mean(leaf_values >= 0.50) * 100.0)

                label = self.labels[idx]
                crop, disease = self._split_label(label)
                vector = predictions[0].numpy()
                top_indices = np.argsort(vector)[::-1][: min(3, len(self.labels))]

                return {
                    "label": label,
                    "crop": crop,
                    "disease": disease,
                    "class_index": idx,
                    "confidence": float(predictions[0, idx].numpy()),
                    "top_predictions": [{"label": self.labels[int(i)], "probability": float(vector[int(i)])} for i in top_indices],
                    "image_sha256": hashlib.sha256(raw).hexdigest(),
                    "severity_score": severity,
                    "heatmap_coverage_percent": coverage,
                    "heatmap_png": self._overlay(original, cam_np),
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
        heat = Image.fromarray(np.uint8(cam * 255), mode="L").resize(image.size, Image.Resampling.BILINEAR)
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
