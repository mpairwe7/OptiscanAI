"""Tests for FHIR R4 and SMART-on-FHIR Protocol Endpoints."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_smart_configuration_endpoint(client):
    response = client.get("/api/v1/fhir/.well-known/smart-configuration")
    assert response.status_code == 200
    data = response.json()
    assert "authorization_endpoint" in data
    assert "token_endpoint" in data
    assert "launch-ehr" in data["capabilities"]
    assert "patient/DiagnosticReport.write" in data["scopes_supported"]


def test_smart_launch_and_token_exchange(client):
    # 1. Launch handshake
    launch_res = client.get(
        "/api/v1/fhir/smart/launch",
        params={"launch": "mock_launch_token_123", "iss": "https://ugandaemr.health.go.ug/fhir"},
    )
    assert launch_res.status_code == 200
    launch_data = launch_res.json()
    assert launch_data["status"] == "ready"
    assert "patient_context" in launch_data

    # 2. Token exchange
    token_res = client.post(
        "/api/v1/fhir/smart/token",
        json={"grant_type": "authorization_code", "code": "AUTH_CODE_987"},
    )
    assert token_res.status_code == 200
    token_data = token_res.json()
    assert token_data["token_type"] == "Bearer"
    assert token_data["patient"].startswith("PAT-UG-")
    assert "access_token" in token_data
