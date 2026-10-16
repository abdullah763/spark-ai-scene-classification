from __future__ import annotations

import io
import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.inference import ONNXSceneClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(
    title="Spark AI Scene Classification API",
    description="Six-class natural-scene image classification for the Spark AI ML Engineer assessment.",
    version="1.0.0",
)
predictor = ONNXSceneClassifier()


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top3: list[dict]
    latency_ms: float


@app.get("/")
def root():
    return {
        "name": "Spark AI Scene Classification API",
        "status": "ready" if predictor.loaded else "degraded",
        "docs": "/docs",
        "predict": "POST /predict",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if predictor.loaded else "degraded",
        "model_loaded": predictor.loaded,
        "model": "mobilenet_v3_small_onnx",
        "load_error": predictor.load_error,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not predictor.loaded:
        raise HTTPException(status_code=503, detail="Model is not available")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB upload limit")

    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    started = time.perf_counter()
    try:
        result = predictor.predict(image)
    except Exception as exc:
        LOGGER.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from exc
    latency_ms = (time.perf_counter() - started) * 1000
    LOGGER.info(
        "prediction class=%s confidence=%.4f latency_ms=%.2f",
        result["predicted_class"],
        result["confidence"],
        latency_ms,
    )
    return {**result, "latency_ms": round(latency_ms, 2)}
