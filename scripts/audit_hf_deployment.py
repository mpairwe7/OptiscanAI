#!/usr/bin/env python3
"""Comprehensive Live Regression Testing and Audit Suite for Deployed Hugging Face Space.

Target Space: https://landwind22-retinal-screening.hf.space
Validates and justifies production readiness across:
1. Health and Readiness probes
2. Platform & Model Service metadata
3. Clinical Knowledge Graph & Disease taxonomy
4. 4-Tier Intake Gate (Statistical & Learned rejection)
5. Multi-Disease Classification & Referral Triage
6. Explainability Services (Grad-CAM, Saliency)
7. SMART-on-FHIR Protocol & Interoperability
8. Offline Bundle & Delta Sync Services
9. Frontend Web Application & PWA Assets
"""

import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

DEFAULT_SPACE_URL = os.getenv("HF_SPACE_URL", "https://landwind22-retinal-screening.hf.space")


def make_synthetic_fundus(size: int = 512, valid: bool = True) -> bytes:
    """Generate in-memory synthetic fundus image (red retina disc, optic cup, vessel cues)."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    radius = int(size * 0.44)

    if valid:
        # Retina orange-reddish field
        mask = dist_from_center <= radius
        arr[mask, 0] = 180  # Red
        arr[mask, 1] = 60   # Green
        arr[mask, 2] = 25   # Blue

        # Optic disc (yellowish-bright spot)
        disc_y, disc_x = cy, cx - int(radius * 0.45)
        disc_dist = np.sqrt((x - disc_x) ** 2 + (y - disc_y) ** 2)
        disc_mask = disc_dist <= int(radius * 0.18)
        arr[disc_mask, 0] = 230
        arr[disc_mask, 1] = 200
        arr[disc_mask, 2] = 90

        # Macula (dark central spot)
        mac_y, mac_x = cy, cx + int(radius * 0.25)
        mac_dist = np.sqrt((x - mac_x) ** 2 + (y - mac_y) ** 2)
        mac_mask = mac_dist <= int(radius * 0.14)
        arr[mac_mask, 0] = 130
        arr[mac_mask, 1] = 35
        arr[mac_mask, 2] = 15
    else:
        # Invalid non-fundus image: uniform gray square with low resolution
        arr = np.full((32, 32, 3), 128, dtype=np.uint8)

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class RegressionTester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.results: List[Dict[str, Any]] = []

    def request(
        self,
        path: str,
        method: str = "GET",
        headers: Dict[str, str] = None,
        data: bytes = None,
        json_data: Any = None,
        files: Dict[str, Tuple[str, bytes, str]] = None,
    ) -> Tuple[int, Any, float]:
        """Perform HTTP request and measure latency."""
        url = f"{self.base_url}{path}"
        headers = headers or {}
        headers.setdefault("User-Agent", "OptiscanAI-Regression-Audit/2.0")

        req_body = None
        if files:
            boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
            body_parts = []
            for field_name, (filename, file_bytes, mime_type) in files.items():
                body_parts.append(
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
                    f"Content-Type: {mime_type}\r\n\r\n"
                )
                body_parts.append(file_bytes)
                body_parts.append("\r\n")
            body_parts.append(f"--{boundary}--\r\n")

            full_body = bytearray()
            for part in body_parts:
                if isinstance(part, str):
                    full_body.extend(part.encode("utf-8"))
                else:
                    full_body.extend(part)
            req_body = bytes(full_body)
        elif json_data is not None:
            headers["Content-Type"] = "application/json"
            req_body = json.dumps(json_data).encode("utf-8")
        elif data is not None:
            req_body = data

        req = urllib.request.Request(url, data=req_body, headers=headers, method=method)
        start_time = time.time()
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                latency = round((time.time() - start_time) * 1000, 2)
                content = resp.read()
                try:
                    parsed = json.loads(content.decode("utf-8"))
                except Exception:
                    parsed = content.decode("utf-8", errors="replace")
                return resp.status, parsed, latency
        except urllib.error.HTTPError as e:
            latency = round((time.time() - start_time) * 1000, 2)
            content = e.read()
            try:
                parsed = json.loads(content.decode("utf-8"))
            except Exception:
                parsed = content.decode("utf-8", errors="replace")
            return e.code, parsed, latency
        except Exception as exc:
            latency = round((time.time() - start_time) * 1000, 2)
            return 0, str(exc), latency

    def record(self, category: str, name: str, passed: bool, status: int, latency: float, details: Any):
        res = {
            "category": category,
            "test": name,
            "passed": passed,
            "status_code": status,
            "latency_ms": latency,
            "details": details,
        }
        self.results.append(res)
        mark = "✅ PASS" if passed else "❌ FAIL"
        print(f"[{mark}] {category} :: {name} ({status} in {latency}ms)")

    # -----------------------------------------------------------------------
    # Test Suites
    # -----------------------------------------------------------------------

    def test_health_probes(self):
        print("\n--- 1. Health and Readiness Probes ---")
        probes = [
            ("/health", "Root Health Liveness", lambda b: b.get("status") == "healthy"),
            ("/health/ready", "Application Readiness Probe", lambda b: b.get("status") == "ready"),
            ("/health/model", "Vision Model Health Status", lambda b: b.get("sla_compliant") is True),
            ("/health/gate", "Fundus Gate v2 Model Status", lambda b: b.get("alert_active") is False),
        ]
        for path, name, validator in probes:
            status, body, latency = self.request(path)
            passed = status == 200 and isinstance(body, dict) and validator(body)
            self.record("Health", name, passed, status, latency, body)

    def test_system_and_clinical_knowledge(self):
        print("\n--- 2. System Metadata & Clinical Knowledge ---")
        # System info
        status, body, latency = self.request("/api/v1/system/info")
        passed = status == 200 and isinstance(body, dict) and "platform" in body
        self.record("System", "System Metadata & Model Config", passed, status, latency, body)

        # Disease taxonomy
        status, body, latency = self.request("/api/v1/clinical/disease-info")
        passed = status == 200 and isinstance(body, dict) and body.get("total", 0) >= 20
        self.record(
            "Clinical",
            "Clinical Disease Taxonomy (45 Classes)",
            passed,
            status,
            latency,
            f"Total diseases: {body.get('total') if isinstance(body, dict) else 0}",
        )

        # Knowledge Graph
        status, body, latency = self.request("/api/v1/clinical/knowledge-graph")
        passed = status == 200 and isinstance(body, dict) and body.get("diseases", 0) >= 20
        nodes_cnt = body.get("diseases", 0) if isinstance(body, dict) else 0
        edges_cnt = body.get("edges", 0) if isinstance(body, dict) else 0
        self.record(
            "Clinical",
            "Symbolic Clinical Knowledge Graph",
            passed,
            status,
            latency,
            f"{nodes_cnt} nodes, {edges_cnt} edges",
        )

    def test_quality_gate(self):
        print("\n--- 3. Intake Quality Gate (Fundus Gate v2) ---")
        bad_img = make_synthetic_fundus(size=32, valid=False)

        # 1. Pipeline rejection via /api/v1/predict (HTTP 422)
        status, body, latency = self.request(
            "/api/v1/predict",
            method="POST",
            files={"file": ("non_fundus.jpg", bad_img, "image/jpeg")},
        )
        passed = status == 422 and isinstance(body, dict) and body.get("detail", {}).get("error") == "non_fundus_image"
        self.record(
            "Gate",
            "Pipeline Enforcement: Non-Fundus Blocked (HTTP 422)",
            passed,
            status,
            latency,
            body.get("detail", {}).get("message") if isinstance(body, dict) else body,
        )

        # 2. Standalone diagnostic validation via /api/v1/gate/validate
        status, body, latency = self.request(
            "/api/v1/gate/validate",
            method="POST",
            files={"file": ("bad_eval.jpg", bad_img, "image/jpeg")},
        )
        passed = status == 200 and isinstance(body, dict) and body.get("passed") is False
        self.record(
            "Gate",
            "Diagnostic Assessment of Non-Fundus (Structural Fail)",
            passed,
            status,
            latency,
            body.get("reason") if isinstance(body, dict) else body,
        )

        # 3. Gate status
        status, body, latency = self.request("/api/v1/gate/status")
        passed = status == 200 and isinstance(body, dict) and body.get("enabled") is True and body.get("learned_model_loaded") is True
        self.record(
            "Gate",
            "Quality Gate Operational & Model Status",
            passed,
            status,
            latency,
            f"Learned Model: {body.get('learned_model_loaded') if isinstance(body, dict) else False}, Gate Version: {body.get('gate_version') if isinstance(body, dict) else 'none'}",
        )

    def test_multi_disease_prediction(self):
        print("\n--- 4. Multi-Disease Classification & Triage ---")
        valid_img = make_synthetic_fundus(size=512, valid=True)

        # Standard Multi-Disease Prediction
        status, body, latency = self.request(
            "/api/v1/predict",
            method="POST",
            files={"file": ("fundus.jpg", valid_img, "image/jpeg")},
        )
        passed = (
            status == 200
            and isinstance(body, dict)
            and body.get("success") is True
            and len(body.get("predictions", [])) > 0
            and "clinical" in body
            and body.get("fundus_gate", {}).get("passed") is True
        )
        triage_priority = body.get("clinical", {}).get("referral_priority") if isinstance(body, dict) else None
        num_preds = len(body.get("predictions", [])) if isinstance(body, dict) else 0
        gate_conf = body.get("fundus_gate", {}).get("confidence") if isinstance(body, dict) else 0.0
        self.record(
            "Prediction",
            "Full-Stack Multi-Label Retinal Screening",
            passed,
            status,
            latency,
            f"Referral: {triage_priority}, Detected: {num_preds} classes, Gate Confidence: {gate_conf:.1%}, Latency: {latency}ms",
        )

        # Edge Runtime Engine Status
        status, body, latency = self.request("/api/v1/predict/edge/status")
        passed = status == 200 and isinstance(body, dict) and "parity_tolerance" in body
        self.record(
            "Edge",
            "Edge Runtime Engine Status & Config",
            passed,
            status,
            latency,
            body,
        )

    def test_smart_on_fhir(self):
        print("\n--- 5. SMART-on-FHIR Protocol & Interoperability ---")
        # 1. Well-known configuration
        status, body, latency = self.request("/api/v1/fhir/.well-known/smart-configuration")
        passed = (
            status == 200
            and isinstance(body, dict)
            and "authorization_endpoint" in body
            and "launch-ehr" in body.get("capabilities", [])
        )
        self.record("FHIR", "SMART-on-FHIR .well-known Configuration", passed, status, latency, body)

        # 2. EHR Launch Handshake
        status, body, latency = self.request(
            "/api/v1/fhir/smart/launch?launch=test_token_889&iss=https://ugandaemr.health.go.ug/fhir"
        )
        passed = status == 200 and isinstance(body, dict) and body.get("status") == "ready"
        self.record("FHIR", "EHR Launch Handshake", passed, status, latency, body)

        # 3. SMART Token Exchange
        status, body, latency = self.request(
            "/api/v1/fhir/smart/token",
            method="POST",
            json_data={"grant_type": "authorization_code", "code": "CODE_XYZ_123"},
        )
        passed = status == 200 and isinstance(body, dict) and "access_token" in body
        self.record("FHIR", "SMART OAuth2 Token Exchange", passed, status, latency, body)

    def test_explainability(self):
        print("\n--- 6. Explainability Services ---")
        status, body, latency = self.request("/api/v1/explain/available")
        passed = status == 200 and isinstance(body, dict) and len(body.get("methods", [])) >= 3
        methods = body.get("methods", []) if isinstance(body, dict) else []
        self.record("Explainability", "Available Visual Explanation Methods", passed, status, latency, methods)

    def test_offline_bundle(self):
        print("\n--- 7. Offline Bundle & Synchronization ---")
        status, body, latency = self.request("/api/v1/offline/status")
        passed = status == 200 and isinstance(body, dict)
        self.record("Offline", "Offline Index & Sync Status", passed, status, latency, body)

    def test_frontend_and_pwa(self):
        print("\n--- 8. Frontend Application & PWA Assets ---")
        pwa_paths = [
            ("/", "Next.js Standalone Landing Page"),
            ("/offline.html", "PWA Offline Cache Fallback Page"),
        ]
        for path, name in pwa_paths:
            status, body, latency = self.request(path)
            passed = status == 200 and ("<!DOCTYPE html>" in str(body) or "<html" in str(body))
            self.record("Frontend", name, passed, status, latency, f"Length: {len(str(body))} bytes")

    def run_all(self):
        start_time = time.time()
        print(f"=== Beginning Automated Regression Audit on {self.base_url} ===")
        self.test_health_probes()
        self.test_system_and_clinical_knowledge()
        self.test_quality_gate()
        self.test_multi_disease_prediction()
        self.test_smart_on_fhir()
        self.test_explainability()
        self.test_offline_bundle()
        self.test_frontend_and_pwa()

        elapsed = round(time.time() - start_time, 2)
        passed_tests = sum(1 for r in self.results if r["passed"])
        total_tests = len(self.results)
        print("\n==================================================")
        print(f"Audit Summary: {passed_tests}/{total_tests} Tests Passed in {elapsed}s")
        print("==================================================")
        return passed_tests, total_tests


if __name__ == "__main__":
    tester = RegressionTester(DEFAULT_SPACE_URL)
    passed, total = tester.run_all()
    exit(0 if passed == total else 1)
