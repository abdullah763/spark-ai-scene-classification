from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CFG
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LOGGER = logging.getLogger(__name__)


class ONNXSceneClassifier:
    def __init__(self, model_path: str | Path | None = None, classes_path: str | Path | None = None):
        default_model = CFG.models_dir / "best_model.onnx"
        default_classes = CFG.models_dir / "class_names.json"
        self.model_path = Path(model_path or os.getenv("MODEL_PATH", default_model))
        self.classes_path = Path(classes_path or os.getenv("CLASS_NAMES_PATH", default_classes))
        self.session = None
        self.class_names: list[str] = []
        self.loaded = False
        self.load_error: str | None = None

        try:
            self._load()
        except Exception as exc:  # API should start even if artifacts are not ready yet.
            self.load_error = str(exc)
            LOGGER.warning("Model not loaded: %s", exc)

    def _load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        if not self.classes_path.exists():
            raise FileNotFoundError(f"Class names not found: {self.classes_path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is not installed") from exc

        self.class_names = json.loads(self.classes_path.read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.loaded = True
        self.load_error = None

    @staticmethod
    def _preprocess(image: Image.Image, image_size: int = 224) -> np.ndarray:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(IMAGENET_STD, dtype=np.float32)
        array = (array - mean) / std
        array = np.transpose(array, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(array, dtype=np.float32)

    def predict(self, image: Image.Image) -> dict:
        if not self.loaded or self.session is None:
            raise RuntimeError(self.load_error or "Model is not loaded")

        tensor = self._preprocess(image)
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        order = np.argsort(probabilities)[::-1]
        best = int(order[0])
        top3 = [
            {"class": self.class_names[int(i)], "confidence": float(probabilities[int(i)])}
            for i in order[:3]
        ]
        return {
            "predicted_class": self.class_names[best],
            "confidence": float(probabilities[best]),
            "top3": top3,
        }
