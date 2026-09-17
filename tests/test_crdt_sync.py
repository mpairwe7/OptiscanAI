"""Tests for CRDT Version Vectors and Offline Delta Sync."""

import time

from backend.app.offline_sync import (
    ChunkDelta,
    DeltaResult,
    DeltaSyncEngine,
    VersionVector,
)


def test_version_vector_causality_and_dominance():
    v1 = VersionVector()
    v1.increment("node_A")
    v1.increment("node_A")

    v2 = VersionVector()
    v2.increment("node_A")

    # v1 dominates v2 because node_A has clock 2 vs 1
    assert v1.dominates(v2)
    assert not v2.dominates(v1)
    assert not v1.is_concurrent(v2)


def test_version_vector_concurrency_and_merge():
    # Node A and Node B edit independently offline
    va = VersionVector(clock={"node_A": 2, "node_B": 1}, timestamp=100.0)
    vb = VersionVector(clock={"node_A": 1, "node_B": 3}, timestamp=105.0)

    assert va.is_concurrent(vb)
    assert vb.is_concurrent(va)

    merged = va.merge(vb)
    assert merged.clock == {"node_A": 2, "node_B": 3}
    assert merged.timestamp == 105.0


def test_version_vector_lww_resolution():
    va = VersionVector(clock={"node_A": 2}, timestamp=100.0)
    vb = VersionVector(clock={"node_B": 1}, timestamp=150.0)

    val_a = "older_data"
    val_b = "newer_data"

    winner_val, resolved_vector = va.resolve_lww(vb, val_a, val_b)
    assert winner_val == "newer_data"
    assert resolved_vector.clock == {"node_A": 2, "node_B": 1}


def test_delta_sync_engine_manifest_preserves_vector_clocks(tmp_path):
    index_dir = tmp_path / "index"
    source_dir = tmp_path / "source"
    index_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)

    engine = DeltaSyncEngine(index_dir=str(index_dir), source_dir=str(source_dir))

    delta = DeltaResult(
        additions=[
            ChunkDelta(
                chunk_id="chunk_1",
                action="add",
                new_hash="hash_123",
                version_vector=VersionVector(clock={"edge_1": 1}, timestamp=time.time()),
            )
        ]
    )

    source_passages = [{"id": "chunk_1", "text": "Sample clinical guidelines"}]
    success = engine.apply_delta(delta, source_passages=source_passages)
    assert success

    # Verify vector clock persisted
    assert "chunk_1" in engine._vector_clocks
    assert engine._vector_clocks["chunk_1"].clock.get("edge_1") == 1
    assert engine._vector_clocks["chunk_1"].clock.get(engine._node_id) == 1

    # Reload engine from disk
    reloaded = DeltaSyncEngine(index_dir=str(index_dir), source_dir=str(source_dir))
    assert "chunk_1" in reloaded._vector_clocks
    assert reloaded._vector_clocks["chunk_1"].clock.get("edge_1") == 1
