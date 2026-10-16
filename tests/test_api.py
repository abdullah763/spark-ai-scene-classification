from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api_main


class FakePredictor:
    loaded = True
    load_error = None

    def predict(self, image):
        return {
            "predicted_class": "forest",
            "confidence": 0.91,
            "top3": [
                {"class": "forest", "confidence": 0.91},
                {"class": "mountain", "confidence": 0.05},
                {"class": "sea", "confidence": 0.02},
            ],
        }


@pytest.fixture(autouse=True)
def fake_predictor(monkeypatch):
    monkeypatch.setattr(api_main, "predictor", FakePredictor())


@pytest.fixture
def client():
    return TestClient(api_main.app)


def make_jpeg_bytes():
    image = Image.new("RGB", (32, 32), (120, 180, 90))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_valid_prediction(client):
    response = client.post(
        "/predict",
        files={"file": ("sample.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "forest"
    assert 0 <= body["confidence"] <= 1
    assert len(body["top3"]) == 3
    assert body["latency_ms"] >= 0


def test_unsupported_content_type(client):
    response = client.post(
        "/predict",
        files={"file": ("sample.txt", b"abc", "text/plain")},
    )
    assert response.status_code == 415


def test_empty_upload(client):
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400


def test_invalid_image(client):
    response = client.post(
        "/predict",
        files={"file": ("bad.jpg", b"not an image", "image/jpeg")},
    )
    assert response.status_code == 400


def test_missing_model_returns_503(client, monkeypatch):
    fake = FakePredictor()
    fake.loaded = False
    fake.load_error = "missing"
    monkeypatch.setattr(api_main, "predictor", fake)
    response = client.post(
        "/predict",
        files={"file": ("sample.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 503
