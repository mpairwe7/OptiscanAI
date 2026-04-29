"""Tests for FastAPI endpoints using TestClient."""
import sys
import io

sys.path.insert(0, "/home/developer/Mpairwe7/MLOPS_V1")

import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient that triggers lifespan events."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def synthetic_jpeg():
    """Create a synthetic fundus-like JPEG image in-memory for upload.

    Uses a red-dominant circle on dark background with sharp boundary
    to pass the fundus gate v2 statistical checks.
    """
    size = 224
    arr = np.zeros((size, size, 3), dtype=np.float32)
    cy, cx = size / 2.0, size / 2.0
    y, x = np.mgrid[0:size, 0:size]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) / (size / 2.0)
    boundary = np.clip(1.0 - (dist - 0.82) * 20, 0, 1)
    arr[:, :, 0] = boundary * 0.55
    arr[:, :, 1] = boundary * 0.30
    arr[:, :, 2] = boundary * 0.12
    center_glow = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * (size * 0.15) ** 2))
    arr[:, :, 0] += center_glow * 0.15 * boundary
    arr[:, :, 1] += center_glow * 0.12 * boundary
    rng = np.random.RandomState(42)
    arr[:, :, 1] += rng.randn(size, size).astype(np.float32) * 0.015 * boundary
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

def test_root_endpoint(client):
    """GET / should return 200 with the app name."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "app" in data
    assert isinstance(data["app"], str)
    assert len(data["app"]) > 0


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    """GET /health should return 200 with status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_health_model_endpoint(client):
    """GET /health/model should return latency and SLA info."""
    response = client.get("/health/model")
    assert response.status_code == 200
    data = response.json()
    assert "sla_compliant" in data
    assert "total_predictions" in data


# ---------------------------------------------------------------------------
# Predict - error cases
# ---------------------------------------------------------------------------

def test_predict_no_file(client):
    """POST /api/v1/predict without a file should return 422."""
    response = client.post("/api/v1/predict")
    assert response.status_code == 422


def test_predict_invalid_file(client):
    """POST /api/v1/predict with a non-image file should return 400."""
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Diseases list endpoint
# ---------------------------------------------------------------------------

def test_diseases_endpoint(client):
    """GET /api/v1/diseases should return a list of diseases."""
    response = client.get("/api/v1/diseases")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "diseases" in data
    assert isinstance(data["diseases"], list)
    assert data["total"] > 0
    # Each disease entry should have code and name
    for disease in data["diseases"]:
        assert "code" in disease
        assert "name" in disease


# ---------------------------------------------------------------------------
# Predict - valid image
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_predict_with_image(client, synthetic_jpeg, monkeypatch):
    """POST /api/v1/predict with a valid JPEG should return 200 with predictions.

    Uses monkeypatch to bypass the v2 gate (which has untrained learned weights
    in test). Gate behavior is tested separately in test_fundus_gate_v2.py.
    """
    from src.data.fundus_gate_v2 import GateResultV2
    monkeypatch.setattr(
        "src.data.fundus_gate_v2.gate_image",
        lambda img: GateResultV2(
            passed=True, confidence=0.95, reason="mocked", checks={},
            layer="fusion", gate_version="v2", fused_confidence=0.95,
            statistical_confidence=0.95,
        ),
    )
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", synthetic_jpeg, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "predictions" in data
    assert "total_detected" in data
    assert "all_probabilities" in data
    assert "inference_ms" in data
    assert isinstance(data["predictions"], list)
    assert isinstance(data["inference_ms"], (int, float))


@pytest.mark.slow
def test_predict_with_threshold(client, synthetic_jpeg, monkeypatch):
    """Prediction threshold parameter should be respected."""
    from src.data.fundus_gate_v2 import GateResultV2
    monkeypatch.setattr(
        "src.data.fundus_gate_v2.gate_image",
        lambda img: GateResultV2(
            passed=True, confidence=0.95, reason="mocked", checks={},
            layer="fusion", gate_version="v2", fused_confidence=0.95,
            statistical_confidence=0.95,
        ),
    )
    synthetic_jpeg.seek(0)
    response = client.post(
        "/api/v1/predict?threshold=0.99",
        files={"file": ("test.jpg", synthetic_jpeg, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["threshold"] == 0.99


# ---------------------------------------------------------------------------
# Gate v2 endpoints
# ---------------------------------------------------------------------------


def test_gate_status_endpoint(client):
    """GET /api/v1/gate/status should return gate info."""
    response = client.get("/api/v1/gate/status")
    assert response.status_code == 200
    data = response.json()
    assert data["gate_version"] == "v2"
    assert "enabled" in data
    assert "config" in data
    assert "stats" in data


def test_gate_validate_endpoint(client):
    """POST /api/v1/gate/validate should always return 200 with full breakdown."""
    # Use a random image (non-fundus) — should still return 200
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/v1/gate/validate",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "passed" in data
    assert "confidence" in data
    assert "layer" in data
    assert "gate_version" in data
    assert data["gate_version"] == "v2"


def test_predict_gate_v2_rejection(client):
    """Non-fundus image should get 422 with v2 gate fields."""
    arr = np.full((224, 224, 3), [0, 0, 255], dtype=np.uint8)  # Solid blue
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/v1/predict",
        files={"file": ("blue.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 422
    data = response.json()["detail"]
    assert data["error"] == "non_fundus_image"
    assert "confidence" in data
    assert "layer" in data


def test_health_gate_endpoint(client):
    """GET /health/gate should return gate metrics."""
    response = client.get("/health/gate")
    assert response.status_code == 200
    data = response.json()
    assert "total_checked" in data
    assert "pass_rate" in data
