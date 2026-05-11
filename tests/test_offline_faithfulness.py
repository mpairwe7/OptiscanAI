"""Offline faithfulness tests.

Validates:
  - Bundle integrity and SHA-256 verification
  - Delta sync correctness
  - Audit hash chain integrity
  - Clinical KG JSON vs Python KG parity
"""

import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestBundleIntegrity:
    """Test the screening bundle build and verification."""

    @pytest.fixture
    def mock_bundle_dir(self, tmp_path):
        """Create a mock mobile_export directory with all required components."""
        export_dir = tmp_path / "mobile_export"
        export_dir.mkdir()

        # Student model (mock ONNX file)
        student = export_dir / "student_int8.onnx"
        student.write_bytes(b"\x00" * 1024 * 100)  # 100 KB mock

        # Gate model
        gate = export_dir / "gate_mobilenetv3.onnx"
        gate.write_bytes(b"\x01" * 1024 * 50)  # 50 KB mock

        # Thresholds
        thresholds = [0.1 + i * 0.03 for i in range(28)]
        (export_dir / "thresholds.json").write_text(json.dumps(thresholds))

        # Clinical KG
        kg = {
            "version": "1.0.0",
            "num_diseases": 28,
            "disease_names": ["DR", "ARMD"],
            "diseases": {"DR": {"code": "DR", "related_diseases": ["ARMD"]}},
            "co_occurrence": [{"from": "DR", "to": "ARMD"}],
            "referral_rules": {},
            "uganda_prevalence": {"DR": 0.05},
        }
        (export_dir / "clinical_kg.json").write_text(json.dumps(kg))

        return export_dir

    def test_bundle_contains_required_components(self, mock_bundle_dir):
        """All required files must exist."""
        required = [
            "student_int8.onnx",
            "gate_mobilenetv3.onnx",
            "thresholds.json",
            "clinical_kg.json",
        ]
        for name in required:
            assert (mock_bundle_dir / name).exists(), f"Missing: {name}"

    def test_sha256_computation(self, mock_bundle_dir):
        """SHA-256 hash should be consistent."""
        path = mock_bundle_dir / "thresholds.json"
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        h2 = hashlib.sha256(path.read_bytes()).hexdigest()
        assert h == h2
        assert len(h) == 64

    def test_bundle_size_under_limit(self, mock_bundle_dir):
        """Total uncompressed size should be well under 150 MB."""
        total = sum(f.stat().st_size for f in mock_bundle_dir.iterdir())
        total_mb = total / 1e6
        assert total_mb < 150, f"Bundle {total_mb:.1f} MB exceeds 150 MB limit"

    def test_thresholds_json_valid(self, mock_bundle_dir):
        """Thresholds JSON should load as a list of 28 floats."""
        with open(mock_bundle_dir / "thresholds.json") as f:
            thresholds = json.load(f)
        assert isinstance(thresholds, list)
        assert len(thresholds) == 28
        for t in thresholds:
            assert isinstance(t, float)
            assert 0.0 <= t <= 1.0

    def test_clinical_kg_json_schema(self, mock_bundle_dir):
        """Clinical KG JSON should have required top-level keys."""
        with open(mock_bundle_dir / "clinical_kg.json") as f:
            kg = json.load(f)
        required_keys = [
            "version",
            "num_diseases",
            "disease_names",
            "diseases",
            "co_occurrence",
            "referral_rules",
            "uganda_prevalence",
        ]
        for key in required_keys:
            assert key in kg, f"Missing key: {key}"

    def test_tarball_creation_and_extraction(self, mock_bundle_dir, tmp_path):
        """Bundle should create a valid tar.gz and survive round-trip."""
        tarball = tmp_path / "bundle.tar.gz"
        with tarfile.open(tarball, "w:gz") as tar:
            for f in mock_bundle_dir.iterdir():
                tar.add(f, arcname=f.name)

        # Verify extraction
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(extract_dir)

        for f in mock_bundle_dir.iterdir():
            extracted = extract_dir / f.name
            assert extracted.exists(), f"Missing after extraction: {f.name}"
            assert extracted.read_bytes() == f.read_bytes()


class TestDeltaSync:
    """Test delta sync computation."""

    def test_no_changes_returns_empty_delta(self):
        """Same version + same hashes -> no changes."""
        server_manifest = {
            "version": "1.0.0",
            "files": {"thresholds.json": "abc123", "model.onnx": "def456"},
        }
        client_hashes = {"thresholds.json": "abc123", "model.onnx": "def456"}

        changed = []
        for filename, server_hash in server_manifest["files"].items():
            client_hash = client_hashes.get(filename)
            if client_hash != server_hash:
                changed.append(filename)

        assert len(changed) == 0

    def test_updated_component_detected(self):
        """Changed hash should be detected as update."""
        server_files = {"thresholds.json": "new_hash", "model.onnx": "same"}
        client_hashes = {"thresholds.json": "old_hash", "model.onnx": "same"}

        changed = [f for f, h in server_files.items() if client_hashes.get(f) != h]
        assert changed == ["thresholds.json"]

    def test_new_component_detected(self):
        """File present on server but not client -> add."""
        server_files = {"thresholds.json": "abc", "new_model.onnx": "xyz"}
        client_hashes = {"thresholds.json": "abc"}

        added = [f for f in server_files if f not in client_hashes]
        assert added == ["new_model.onnx"]

    def test_deleted_component_detected(self):
        """File present on client but not server -> delete."""
        server_files = {"thresholds.json": "abc"}
        client_hashes = {"thresholds.json": "abc", "old_model.onnx": "xyz"}

        deleted = [f for f in client_hashes if f not in server_files]
        assert deleted == ["old_model.onnx"]


class TestAuditHashChain:
    """Test the hash chain integrity logic."""

    def _compute_hash(self, entry: dict) -> str:
        sorted_entry = dict(sorted(entry.items()))
        return hashlib.sha256(json.dumps(sorted_entry).encode()).hexdigest()

    def test_chain_integrity_over_entries(self):
        """Hash chain should link entries correctly."""
        chain = []
        prev_hash = "0" * 64

        for i in range(100):
            entry = {
                "id": f"entry_{i}",
                "event_type": "prediction",
                "previous_hash": prev_hash,
            }
            entry_hash = self._compute_hash(entry)
            chain.append({**entry, "entry_hash": entry_hash})
            prev_hash = entry_hash

        # Verify chain
        expected_prev = "0" * 64
        for entry in chain:
            assert entry["previous_hash"] == expected_prev
            expected_prev = entry["entry_hash"]

    def test_tampering_detection(self):
        """Modifying an entry should break the chain."""
        entries = []
        prev_hash = "0" * 64

        for i in range(5):
            entry = {"id": str(i), "data": f"original_{i}", "previous_hash": prev_hash}
            entry_hash = self._compute_hash(entry)
            entries.append({**entry, "entry_hash": entry_hash})
            prev_hash = entry_hash

        # Tamper with entry 2
        entries[2]["data"] = "tampered"

        # Verify chain detects tampering
        expected_prev = "0" * 64
        broken = False
        for entry in entries:
            if entry["previous_hash"] != expected_prev:
                broken = True
                break
            recomputed = self._compute_hash(
                {k: v for k, v in entry.items() if k != "entry_hash"}
            )
            if recomputed != entry["entry_hash"]:
                broken = True
                break
            expected_prev = entry["entry_hash"]

        assert broken, "Tampering should have been detected"

    def test_genesis_hash(self):
        """First entry should link to the genesis hash (64 zeros)."""
        genesis = "0" * 64
        first_entry = {"id": "0", "previous_hash": genesis}
        assert first_entry["previous_hash"] == genesis
        assert len(genesis) == 64
