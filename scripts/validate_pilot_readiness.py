#!/usr/bin/env python3
"""Validate "Ready for National Pilot" checklist.

Programmatically checks all readiness criteria for Uganda deployment.

Usage:
    PYTHONPATH=. python scripts/validate_pilot_readiness.py

Produces:
    outputs/pilot_readiness_report.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class PilotReadinessValidator:
    """Validates all readiness criteria for national pilot deployment."""

    def __init__(self):
        self.checks: list[dict] = []

    def _check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        return passed

    def check_model_artifacts(self) -> bool:
        """Verify model files exist."""
        paths = [
            "outputs/mobile_export/student_int8.onnx",
            "outputs/mobile_export/gate_mobilenetv3.onnx",
            "outputs/mobile_export/thresholds.json",
            "outputs/mobile_export/clinical_kg.json",
        ]
        missing = [p for p in paths if not Path(p).exists()]
        return self._check(
            "Model artifacts exist",
            len(missing) == 0,
            f"Missing: {missing}" if missing else "All present",
        )

    def check_bundle_size(self) -> bool:
        """Verify bundle <= 150 MB."""
        bundle_dir = Path("outputs/bundles")
        if not bundle_dir.exists():
            return self._check("Bundle size", False, "No bundles directory")
        bundles = list(bundle_dir.glob("*.tar.gz"))
        if not bundles:
            return self._check("Bundle size", False, "No bundle archives found")
        size_mb = bundles[0].stat().st_size / 1e6
        return self._check("Bundle size <= 150 MB", size_mb <= 150, f"{size_mb:.1f} MB")

    def check_student_model_size(self) -> bool:
        """Verify INT8 model <= 50 MB."""
        path = Path("outputs/mobile_export/student_int8.onnx")
        if not path.exists():
            return self._check("Student model size", False, "File not found")
        size_mb = path.stat().st_size / 1e6
        return self._check("Student model <= 50 MB", size_mb <= 50, f"{size_mb:.1f} MB")

    def check_flutter_app(self) -> bool:
        """Verify Flutter app structure exists."""
        pubspec = Path("mobile/retinalai/pubspec.yaml")
        return self._check("Flutter app exists", pubspec.exists())

    def check_voice_backend(self) -> bool:
        """Verify voice backend files exist."""
        files = [
            "backend/app/routers/voice.py",
            "backend/app/core/voice_pipeline.py",
            "backend/app/core/asr_engine.py",
            "backend/app/core/tts_engine.py",
            "backend/app/core/vad_engine.py",
        ]
        missing = [f for f in files if not Path(f).exists()]
        return self._check("Voice backend", len(missing) == 0, f"Missing: {missing}" if missing else "All present")

    def check_luganda_support(self) -> bool:
        """Verify Luganda language files."""
        files = [
            "backend/app/core/luganda/clinical_terms.py",
            "backend/app/core/luganda/code_switch.py",
        ]
        missing = [f for f in files if not Path(f).exists()]
        return self._check("Luganda support", len(missing) == 0)

    def check_dhis2_integration(self) -> bool:
        """Verify DHIS2 integration module."""
        return self._check(
            "DHIS2 integration",
            Path("backend/app/integrations/dhis2/client.py").exists(),
        )

    def check_privacy_compliance(self) -> bool:
        """Verify PDP Act compliance module."""
        return self._check(
            "PDP Act 2019 compliance",
            Path("backend/app/core/privacy/consent.py").exists(),
        )

    def check_governance_artifacts(self) -> bool:
        """Verify governance stack exists."""
        files = [
            "src/governance/bias_auditor.py",
            "src/governance/model_card.py",
            "src/governance/audit_logger.py",
        ]
        missing = [f for f in files if not Path(f).exists()]
        return self._check("Governance stack", len(missing) == 0)

    def check_tests_exist(self) -> bool:
        """Verify test suite."""
        test_files = list(Path("tests").glob("test_*.py"))
        return self._check("Test suite", len(test_files) >= 15, f"{len(test_files)} test files")

    def check_ci_pipeline(self) -> bool:
        """Verify CI workflow exists."""
        return self._check(
            "CI pipeline",
            Path(".github/workflows/ml-pipeline.yml").exists()
            or Path(".github/workflows/quantization.yml").exists(),
        )

    def check_docker(self) -> bool:
        """Verify Docker deployment files."""
        return self._check(
            "Docker deployment",
            Path("Dockerfile.hf").exists() and Path("docker-compose.yml").exists(),
        )

    def generate_report(self) -> dict:
        """Run all checks and generate the readiness report."""
        self.checks = []

        self.check_model_artifacts()
        self.check_bundle_size()
        self.check_student_model_size()
        self.check_flutter_app()
        self.check_voice_backend()
        self.check_luganda_support()
        self.check_dhis2_integration()
        self.check_privacy_compliance()
        self.check_governance_artifacts()
        self.check_tests_exist()
        self.check_ci_pipeline()
        self.check_docker()

        passed = sum(1 for c in self.checks if c["passed"])
        total = len(self.checks)

        return {
            "ready": passed == total,
            "passed": passed,
            "total": total,
            "checks": self.checks,
        }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    validator = PilotReadinessValidator()
    report = validator.generate_report()

    output_path = Path("outputs/pilot_readiness_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"PILOT READINESS: {'READY' if report['ready'] else 'NOT READY'} ({report['passed']}/{report['total']})")
    print(f"{'=' * 60}")
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        detail = f" — {check['detail']}" if check.get("detail") else ""
        print(f"  [{status}] {check['name']}{detail}")
    print(f"{'=' * 60}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
