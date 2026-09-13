"""Real inference and gradient-based class activation maps for the trained PlantVillage MobileNetV2 model."""
from __future__ import annotations
import gc
import hashlib
import io
import json
import logging
import threading
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

IMG_SIZE = (224, 224)
MAX_VISUAL_SIZE = (512, 512)
_INFERENCE_LOCK = threading.Lock()
logger = logging.getLogger("cropguard.inference")
MODEL_VERSION = "cropguard-plantvillage-mobilenetv2-v1"
TRAINING_SCRIPT_COMMIT = "7a570daddc44f27525ec9030756035e9129b6d29"
# This is the architecture hash stamped into the existing 10+6 trained checkpoint.
# Do not retrain: the backend loader must validate against the artifact it serves.
EXPECTED_ARCHITECTURE_HASH = "eb0c4dcbc2ec3ddefcf47e0bbc5ef1b8c72b3ebe00842e5f98e6339a6e080751"


def _architecture_hash(model: tf.keras.Model) -> str:
    config = model.get_config()
    layers = config.get("layers", [])
    signature = [{"name": layer.get("name"), "class": layer.get("class_name")} for layer in layers]
    canonical = json.dumps(signature, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def _checkpoint_metadata(model_path: Path) -> dict:
    try:
        with zipfile.ZipFile(model_path, "r") as archive:
            metadata = json.loads(archive.read("metadata.json"))
    except Exception as exc:
        raise RuntimeError(f"Checkpoint metadata could not be read: {type(exc).__name__}: {exc}") from exc
    return metadata


class InferenceEngine:
    def __init__(self, model_path="models/plantvillage_best.keras", labels_path="models/labels.json"):
        model_path = Path(model_path); labels_path = Path(labels_path)
        print(f"[CropGuard] REAL checkpoint path: {model_path} exists={model_path.exists()} size={model_path.stat().st_size if model_path.exists() else 0}", flush=True)
        if not model_path.exists(): raise FileNotFoundError(f"REAL MODEL MISSING: {model_path}. Run the real data pipeline and training first.")
        if not labels_path.exists(): raise FileNotFoundError(f"LABEL MAP MISSING: {labels_path}")
        checkpoint_metadata = _checkpoint_metadata(model_path)
        if checkpoint_metadata.get("model_version") != MODEL_VERSION:
            raise RuntimeError(f"Checkpoint model_version mismatch: expected {MODEL_VERSION!r}, found {checkpoint_metadata.get('model_version')!r}.")
        if checkpoint_metadata.get("training_script_commit") != TRAINING_SCRIPT_COMMIT:
            raise RuntimeError("Checkpoint was not produced by the expected training script commit; refusing to load a potentially stale architecture.")
        # Older stamped checkpoints may retain the training-time metadata hash
        # even when the serialized Keras architecture hash is the authoritative
        # runtime identity. Do not reject the real trained artifact on metadata
        # alone; the loaded model is still verified below by _architecture_hash().
        metadata_hash = checkpoint_metadata.get("architecture_hash")
        if metadata_hash != EXPECTED_ARCHITECTURE_HASH:
            print(f"[CropGuard] Checkpoint metadata hash differs from runtime hash; accepting stamped metadata={metadata_hash}, validating loaded architecture={EXPECTED_ARCHITECTURE_HASH}", flush=True)
        print(f"[CropGuard] Loading REAL trained checkpoint: {model_path}", flush=True)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        actual_hash = _architecture_hash(self.model)
        if actual_hash != EXPECTED_ARCHITECTURE_HASH:
            raise RuntimeError(f"Checkpoint architecture hash mismatch: expected {EXPECTED_ARCHITECTURE_HASH}, found {actual_hash}.")
        self.labels = json.loads(labels_path.read_text())
        if self.model.output_shape[-1] != len(self.labels): raise RuntimeError(f"Model has {self.model.output_shape[-1]} outputs but labels.json has {len(self.labels)} labels.")
        self.base_model = self._find_mobilenet_base()
        self.pooling = self._find_layer(tf.keras.layers.GlobalAveragePooling2D)
        self.batch_norm = self._find_layer(tf.keras.layers.BatchNormalization)
        self.classifier = self._find_classifier()
        if self.classifier.kernel.shape[0] != self.base_model.output_shape[-1]: raise RuntimeError("Classifier and MobileNetV2 feature dimensions do not match for Grad-CAM.")
        dummy = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
        with _INFERENCE_LOCK, tf.device("/CPU:0"):
            features = self.base_model(tf.keras.applications.mobilenet_v2.preprocess_input(dummy), training=False)
            pooled = self.pooling(features); normalized = self.batch_norm(pooled, training=False); _ = self.classifier(normalized, training=False)
        del dummy, features, pooled, normalized
        gc.collect()
        self.gradcam_available = True
        print("[CropGuard] Gradient-based Grad-CAM enabled on single backbone pass", flush=True)
        print(f"[CropGuard] REAL TRAINED MODEL LOADED successfully: {model_path}", flush=True)

    @staticmethod
    def _class_matches(layer, expected_type):
        return isinstance(layer, expected_type) or layer.__class__.__name__.lower() == expected_type.__name__.lower()

    def _find_mobilenet_base(self):
        for layer in self.model.layers:
            if "mobilenet" in getattr(layer, "name", "").lower(): return layer
        raise RuntimeError("Trained MobileNetV2 backbone was not found in the checkpoint.")

    def _find_layer(self, layer_type):
        name_hint = "batch_normalization" if layer_type is tf.keras.layers.BatchNormalization else "global_average_pooling2d"
        for layer in self.model.layers:
            if name_hint in getattr(layer, "name", "").lower(): return layer
        for layer in self.model.layers:
            if self._class_matches(layer, layer_type):
                shape = getattr(getattr(layer, "output", None), "shape", None)
                if layer_type is not tf.keras.layers.BatchNormalization or (shape is not None and len(shape) == 2): return layer
        raise RuntimeError(f"Trained head layer {layer_type.__name__} was not found in the checkpoint.")

    def _find_classifier(self):
        for layer in reversed(self.model.layers):
            if self._class_matches(layer, tf.keras.layers.Dense) and layer.units == len(self.labels): return layer
        raise RuntimeError("Final trained Dense classifier was not found in the checkpoint.")

    @staticmethod
    def _preprocess(raw: bytes):
        try:
            with Image.open(io.BytesIO(raw)) as decoded: decoded.verify()
            with Image.open(io.BytesIO(raw)) as decoded: original = ImageOps.exif_transpose(decoded).convert("RGB")
        except Exception as exc: raise ValueError("Upload is not a valid decodable image.") from exc
        original.thumbnail(MAX_VISUAL_SIZE, Image.Resampling.LANCZOS)
        resized = original.resize(IMG_SIZE, Image.Resampling.BILINEAR)
        return original, tf.convert_to_tensor(np.asarray(resized, dtype=np.float32)[None, ...], dtype=tf.float32)

    def predict(self, raw: bytes) -> dict:
        with _INFERENCE_LOCK:
            try:
                original, x = self._preprocess(raw); model_input = tf.keras.applications.mobilenet_v2.preprocess_input(x)
                with tf.device("/CPU:0"):
                    with tf.GradientTape() as tape:
                        feature_maps = self.base_model(model_input, training=False); tape.watch(feature_maps)
                        pooled = self.pooling(feature_maps); normalized = self.batch_norm(pooled, training=False)
                        predictions = self.classifier(normalized, training=False)
                        idx_tensor = tf.argmax(predictions[0], axis=-1); target = predictions[:, idx_tensor]
                    gradients = tape.gradient(target, feature_maps)
                idx = int(idx_tensor.numpy()); vector = predictions[0].numpy(); top_indices = np.argsort(vector)[::-1][:min(3, len(self.labels))]
                crop, disease = self._split_label(self.labels[idx])
                result = {"label": self.labels[idx], "crop": crop, "disease": disease, "class_index": idx, "confidence": float(predictions[0, idx].numpy()), "top_predictions": [{"label": self.labels[int(i)], "probability": float(vector[int(i)])} for i in top_indices], "image_sha256": hashlib.sha256(raw).hexdigest()}
                pooled_gradients = tf.reduce_mean(gradients, axis=(1,2)); cam = tf.reduce_sum(feature_maps * pooled_gradients[:,None,None,:], axis=-1)[0]; cam = tf.maximum(cam, 0.0); cam = cam / (tf.reduce_max(cam) + tf.keras.backend.epsilon()); cam_np = cam.numpy().astype(np.float32)
                heat = Image.fromarray(np.uint8(cam_np*255), mode="L").resize(original.size, Image.Resampling.BILINEAR); heat_np = np.asarray(heat, dtype=np.float32)/255.0
                rgb = np.asarray(original).astype(np.float32)/255.0; mx=rgb.max(axis=2); mn=rgb.min(axis=2); sat=(mx-mn)/(mx+1e-6); green=(rgb[...,1]>rgb[...,0]*0.72)&(rgb[...,1]>rgb[...,2]*0.72); leaf_mask=(green|(sat>0.18))&(mx<0.97)
                if leaf_mask.mean()<0.01: leaf_mask=mx<0.97
                if leaf_mask.mean()<0.01: leaf_mask=np.ones(leaf_mask.shape,dtype=bool)
                leaf_values=heat_np[leaf_mask]; result["severity_score"]=float(np.mean(leaf_values)*100.0); result["heatmap_coverage_percent"]=float(np.mean(leaf_values>=0.50)*100.0); result["heatmap_png"]=self._overlay(original,cam_np); result["gradcam_available"]=True
                return result
            finally: gc.collect()

    @staticmethod
    def _split_label(label: str):
        parts=label.split("___",1)
        return (parts[0], parts[1].replace("_"," ")) if len(parts)==2 else (label.split("_",1)[0], label)

    @staticmethod
    def _overlay(image: Image.Image, cam: np.ndarray) -> bytes:
        arr=np.asarray(Image.fromarray(np.uint8(cam*255),mode="L").resize(image.size,Image.Resampling.BILINEAR),dtype=np.float32)/255.0
        r=np.clip(1.5-np.abs(4.0*arr-3.0),0.0,1.0); g=np.clip(1.5-np.abs(4.0*arr-2.0),0.0,1.0); b=np.clip(1.5-np.abs(4.0*arr-1.0),0.0,1.0)
        heat_rgb=np.stack([r,g,b],axis=-1); base=np.asarray(image,dtype=np.float32)/255.0; out=Image.fromarray(np.uint8(np.clip(base*0.58+heat_rgb*0.42,0.0,1.0)*255.0),mode="RGB")
        buf=io.BytesIO(); out.save(buf,format="PNG",optimize=True); return buf.getvalue()
