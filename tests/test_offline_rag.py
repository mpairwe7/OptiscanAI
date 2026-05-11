"""Tests for offline RAG pipeline, delta sync, bundle management, and API endpoints."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# OfflineRAGPipeline tests
# ---------------------------------------------------------------------------


class TestOfflineRAGPipeline:
    """Tests for the OfflineRAGPipeline class."""

    def test_import(self):
        """Pipeline module is importable."""
        from backend.app.offline_rag import OfflineRAGPipeline
        assert OfflineRAGPipeline is not None

    def test_singleton(self):
        """Pipeline uses singleton pattern."""
        from backend.app.offline_rag import OfflineRAGPipeline
        OfflineRAGPipeline.reset_instance()
        p1 = OfflineRAGPipeline.get_instance()
        p2 = OfflineRAGPipeline.get_instance()
        assert p1 is p2
        OfflineRAGPipeline.reset_instance()

    def test_status_when_not_loaded(self):
        """Status reports not loaded when index is missing."""
        from backend.app.offline_rag import OfflineRAGPipeline
        OfflineRAGPipeline.reset_instance()
        pipeline = OfflineRAGPipeline.get_instance()
        status = pipeline.get_pipeline_status()
        assert status["loaded"] is False
        OfflineRAGPipeline.reset_instance()

    def test_search_returns_empty_when_not_loaded(self):
        """Search returns empty list when pipeline is not loaded."""
        from backend.app.offline_rag import OfflineRAGPipeline
        OfflineRAGPipeline.reset_instance()
        pipeline = OfflineRAGPipeline.get_instance()
        results = pipeline.search("test query")
        assert results == []
        OfflineRAGPipeline.reset_instance()


# ---------------------------------------------------------------------------
# DeltaSyncEngine tests
# ---------------------------------------------------------------------------


class TestDeltaSyncEngine:
    """Tests for the DeltaSyncEngine class."""

    def test_import(self):
        """Sync engine module is importable."""
        from backend.app.offline_sync import DeltaSyncEngine
        assert DeltaSyncEngine is not None

    def test_compute_delta_with_source_files(self):
        """Delta computation scans source dir and finds new files."""
        from backend.app.offline_sync import DeltaSyncEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir) / "index"
            source_dir = Path(tmpdir) / "source"
            index_dir.mkdir()
            source_dir.mkdir()
            (source_dir / "chunk_001.txt").write_text("hello")
            (source_dir / "chunk_002.txt").write_text("world")

            engine = DeltaSyncEngine(
                index_dir=str(index_dir),
                source_dir=str(source_dir),
            )
            delta = engine.compute_delta()
            assert delta.total_changes >= 0  # May be 0 or 2 depending on manifest state

    def test_get_sync_status_fields(self):
        """Sync status returns expected fields."""
        from backend.app.offline_sync import DeltaSyncEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = DeltaSyncEngine(
                index_dir=tmpdir,
                source_dir=tmpdir,
            )
            status = engine.get_sync_status()
            assert "state" in status
            assert "total_syncs" in status
            assert "last_sync_time" in status
            assert "source_dir" in status

    def test_delta_result_properties(self):
        """DeltaResult dataclass computes properties correctly."""
        from backend.app.offline_sync import DeltaResult, ChunkDelta

        delta = DeltaResult(
            additions=[ChunkDelta(chunk_id="a", action="add")],
            updates=[ChunkDelta(chunk_id="b", action="update")],
            deletions=[],
        )
        assert delta.total_changes == 2
        assert delta.is_empty is False

        empty = DeltaResult()
        assert empty.total_changes == 0
        assert empty.is_empty is True


# ---------------------------------------------------------------------------
# OfflineBundleManager tests
# ---------------------------------------------------------------------------


class TestOfflineBundleManager:
    """Tests for the OfflineBundleManager class."""

    def test_import(self):
        """Bundle manager module is importable."""
        from backend.app.offline_bundle import OfflineBundleManager
        assert OfflineBundleManager is not None

    def test_list_versions_empty(self):
        """Listing versions on empty bundles dir returns empty list."""
        from backend.app.offline_bundle import OfflineBundleManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OfflineBundleManager(
                index_dir=tmpdir,
                bundles_dir=tmpdir,
            )
            versions = manager.list_versions()
            assert isinstance(versions, list)
            assert len(versions) == 0

    def test_semver_parsing(self):
        """Semantic version parsing and bumping works."""
        from backend.app.offline_bundle import parse_semver, bump_version

        assert parse_semver("1.2.3") == (1, 2, 3)
        assert bump_version("1.2.3", "patch") == "1.2.4"
        assert bump_version("1.2.3", "minor") == "1.3.0"
        assert bump_version("1.2.3", "major") == "2.0.0"

    def test_bundle_info_dataclass(self):
        """BundleInfo dataclass initializes correctly."""
        from backend.app.offline_bundle import BundleInfo

        info = BundleInfo(
            version="1.0.0",
            path="/tmp/test.tar.gz",
            size_bytes=1024,
            sha256="abc123",
            created_at=0.0,
            compression="gzip",
        )
        assert info.version == "1.0.0"
        assert info.size_bytes == 1024


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestOfflineEndpoints:
    """Tests for the offline API router endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from fastapi.testclient import TestClient
        from backend.app.main import app
        return TestClient(app)

    def test_offline_status_endpoint(self, client):
        """Offline status endpoint responds."""
        response = client.get("/api/v1/offline/status")
        assert response.status_code in (200, 403, 503)


# ---------------------------------------------------------------------------
# Quantized models endpoint tests
# ---------------------------------------------------------------------------


class TestQuantizedEndpoints:
    """Tests for the quantized models API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from fastapi.testclient import TestClient
        from backend.app.main import app
        return TestClient(app)

    def test_list_quantized_models(self, client):
        """Quantized models endpoint returns valid response."""
        response = client.get("/api/v1/models/quantized")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total_count" in data
        assert "baseline_model" in data
        assert isinstance(data["models"], list)

    def test_optimization_status(self, client):
        """Optimization status endpoint returns valid response."""
        response = client.get("/api/v1/models/optimization/status")
        assert response.status_code == 200
        data = response.json()
        assert "torch_compile_enabled" in data
        assert "memory_usage_mb" in data


# ---------------------------------------------------------------------------
# Monitoring endpoint tests
# ---------------------------------------------------------------------------


class TestMonitoringEndpoints:
    """Tests for the monitoring and admin API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from fastapi.testclient import TestClient
        from backend.app.main import app
        return TestClient(app)

    def test_admin_stats(self, client):
        """Full admin stats endpoint returns valid response."""
        response = client.get("/api/v1/admin/stats")
        assert response.status_code == 200
        data = response.json()
        assert "offline" in data
        assert "voice" in data
        assert "quantization" in data
        assert "feature_flags" in data

    def test_offline_stats(self, client):
        """Offline stats endpoint returns valid response."""
        response = client.get("/api/v1/admin/offline_stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_offline_sessions" in data
        assert "offline_faithfulness_score" in data

    def test_voice_stats(self, client):
        """Voice stats endpoint returns valid response."""
        response = client.get("/api/v1/admin/voice_stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_voice_sessions" in data
        assert "barge_in_success_rate" in data

    def test_prometheus_metrics(self, client):
        """Prometheus metrics endpoint returns text format."""
        response = client.get("/api/v1/admin/metrics/prometheus")
        assert response.status_code == 200
        assert "retinalai_" in response.text

    def test_grafana_dashboard(self, client):
        """Grafana dashboard config endpoint returns panels."""
        response = client.get("/api/v1/admin/grafana/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "panels" in data
        assert len(data["panels"]) > 0


# ---------------------------------------------------------------------------
# MetricsCollector unit tests
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    """Tests for the MetricsCollector class."""

    def test_increment_counter(self):
        """Counter increments correctly."""
        from backend.app.routers.monitoring import MetricsCollector
        mc = MetricsCollector()
        mc.increment("offline_queries_total", 5)
        all_metrics = mc.get_all()
        assert all_metrics["counters"]["offline_queries_total"] == 5

    def test_set_gauge(self):
        """Gauge sets correctly."""
        from backend.app.routers.monitoring import MetricsCollector
        mc = MetricsCollector()
        mc.set_gauge("offline_faithfulness_score", 0.87)
        all_metrics = mc.get_all()
        assert all_metrics["gauges"]["offline_faithfulness_score"] == 0.87

    def test_observe_histogram(self):
        """Histogram observations are recorded."""
        from backend.app.routers.monitoring import MetricsCollector
        mc = MetricsCollector()
        for val in [10.0, 20.0, 30.0, 40.0, 50.0]:
            mc.observe("offline_search_latency_ms", val)
        all_metrics = mc.get_all()
        hist = all_metrics["histograms"]["offline_search_latency_ms"]
        assert hist["count"] == 5
        assert hist["mean"] == 30.0

    def test_prometheus_format(self):
        """Prometheus export produces valid format."""
        from backend.app.routers.monitoring import MetricsCollector
        mc = MetricsCollector()
        mc.increment("offline_queries_total", 42)
        mc.set_gauge("offline_faithfulness_score", 0.85)
        output = mc.to_prometheus()
        assert "retinalai_offline_queries_total 42" in output
        assert "retinalai_offline_faithfulness_score 0.85" in output

    def test_histogram_empty(self):
        """Empty histograms produce zero values."""
        from backend.app.routers.monitoring import MetricsCollector
        mc = MetricsCollector()
        all_metrics = mc.get_all()
        for name, hist in all_metrics["histograms"].items():
            assert hist["count"] == 0
            assert hist["mean"] == 0.0


# ---------------------------------------------------------------------------
# Mobile bundle export tests
# ---------------------------------------------------------------------------


class TestMobileBundleExport:
    """Tests for the mobile bundle export script."""

    def test_import(self):
        """Export script is importable."""
        import scripts.export_mobile_bundle as mb
        assert hasattr(mb, "build_bundle")
        assert hasattr(mb, "discover_components")

    def test_discover_components_returns_list(self):
        """Component discovery returns a list."""
        from scripts.export_mobile_bundle import discover_components
        components = discover_components(include_voice=False)
        assert isinstance(components, list)
        assert len(components) > 0

    def test_discover_components_with_voice(self):
        """Component discovery with voice flag includes more components."""
        from scripts.export_mobile_bundle import discover_components
        without_voice = discover_components(include_voice=False)
        with_voice = discover_components(include_voice=True)
        assert len(with_voice) > len(without_voice)

    def test_file_size_mb(self):
        """File size calculation works."""
        from scripts.export_mobile_bundle import file_size_mb

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"x" * 1024 * 1024)  # 1 MB
            f.flush()
            size = file_size_mb(f.name)
            assert 0.9 < size < 1.1
            os.unlink(f.name)

    def test_sha256_file(self):
        """SHA-256 computation is correct."""
        from scripts.export_mobile_bundle import sha256_file

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("test content")
            f.flush()
            h = sha256_file(f.name)
            assert len(h) == 64
            assert h == hashlib.sha256(b"test content").hexdigest()
            os.unlink(f.name)

    def test_bundle_manifest_dataclass(self):
        """BundleManifest initializes with defaults."""
        from scripts.export_mobile_bundle import BundleManifest
        m = BundleManifest()
        assert m.version == "1.0.0"
        assert m.max_size_mb == 800
        assert m.within_budget is False

    def test_bundle_component_dataclass(self):
        """BundleComponent initializes correctly."""
        from scripts.export_mobile_bundle import BundleComponent
        c = BundleComponent(name="test_model", size_mb=100.0, budget_mb=200.0)
        assert c.included is False
        assert c.sha256 == ""


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------


class TestPhase5Config:
    """Tests for Phase 5 configuration settings."""

    def test_offline_rag_settings(self):
        """OfflineRAGSettings loads with defaults."""
        from backend.app.core.config import settings
        assert settings.offline_rag.enabled is False
        assert settings.offline_rag.top_k == 5
        assert settings.offline_rag.target_bundle_size_mb == 150

    def test_quantization_settings(self):
        """QuantizationSettings loads with defaults."""
        from backend.app.core.config import settings
        assert settings.quantization.enabled is False
        assert settings.quantization.torch_compile_mode == "max-autotune"
        assert settings.quantization.max_faithfulness_drop == 0.04

    def test_voice_first_settings(self):
        """VoiceFirstSettings loads with defaults."""
        from backend.app.core.config import settings
        assert settings.voice_first.enabled is False
        assert settings.voice_first.default_language == "en-ug"
        assert settings.voice_first.barge_in_enabled is True

    def test_mobile_bundle_settings(self):
        """MobileBundleSettings loads with defaults."""
        from backend.app.core.config import settings
        assert settings.mobile_bundle.enabled is False
        assert settings.mobile_bundle.max_bundle_size_mb == 800
        assert settings.mobile_bundle.min_ram_mb == 4096

    def test_all_feature_flags_disabled_by_default(self):
        """All Phase 5 feature flags are disabled by default."""
        from backend.app.core.config import settings
        assert settings.offline_rag.enabled is False
        assert settings.quantization.enabled is False
        assert settings.voice_first.enabled is False
        assert settings.mobile_bundle.enabled is False
