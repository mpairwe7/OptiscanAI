"""Delta Sync Engine for Offline RAG Bundles.

Provides bandwidth-efficient synchronisation of the offline RAG index
by transferring only changed chunks.  Each passage chunk is tracked via
SHA-256 hash in a local JSON manifest so that ``compute_delta()`` can
determine the minimal set of additions, updates, and deletions.

Key features:
- Hash-based (SHA-256) change detection per chunk
- Background async sync via ``asyncio``
- Integrity verification after each delta apply
- Configurable retry with exponential back-off
- P2P sync stub for future mesh-network operation
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class SyncState(str, Enum):
    """Current state of the sync engine."""
    IDLE = "idle"
    SYNCING = "syncing"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class ChunkDelta:
    """Describes a single chunk-level change."""
    chunk_id: str
    action: str  # "add" | "update" | "delete"
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    size_bytes: int = 0


@dataclass
class DeltaResult:
    """Full delta between local and remote manifests."""
    additions: List[ChunkDelta] = field(default_factory=list)
    updates: List[ChunkDelta] = field(default_factory=list)
    deletions: List[ChunkDelta] = field(default_factory=list)
    total_transfer_bytes: int = 0

    @property
    def total_changes(self) -> int:
        return len(self.additions) + len(self.updates) + len(self.deletions)

    @property
    def is_empty(self) -> bool:
        return self.total_changes == 0


@dataclass
class SyncStatus:
    """Snapshot of the sync engine state."""
    state: str = SyncState.IDLE.value
    last_sync_time: Optional[float] = None
    last_sync_duration_s: Optional[float] = None
    last_delta_changes: int = 0
    total_syncs: int = 0
    total_errors: int = 0
    consecutive_errors: int = 0
    local_manifest_chunks: int = 0
    next_sync_time: Optional[float] = None
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _sha256_str(text: str) -> str:
    """Return hex SHA-256 digest for a string."""
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Delta Sync Engine
# ---------------------------------------------------------------------------


class DeltaSyncEngine:
    """Manages bandwidth-efficient delta synchronisation of RAG passages.

    The engine keeps a *local manifest* -- a JSON file mapping each chunk ID
    to its SHA-256 hash.  When ``sync()`` is called it:

    1. Loads the *remote manifest* (from a local source dir or future HTTP).
    2. Computes the delta between local and remote.
    3. Copies only changed chunks into the working index directory.
    4. Updates the local manifest and verifies integrity.

    Usage::

        engine = DeltaSyncEngine()
        await engine.sync()
        status = engine.get_sync_status()
    """

    def __init__(
        self,
        index_dir: Optional[str] = None,
        source_dir: Optional[str] = None,
        manifest_name: str = "sync_manifest.json",
        sync_interval_s: float = 3600.0,
        max_retries: int = 3,
        retry_backoff_base_s: float = 5.0,
    ) -> None:
        self._index_dir = Path(index_dir or os.getenv(
            "OFFLINE_RAG_INDEX_DIR", "data/offline_rag/index",
        ))
        self._source_dir = Path(source_dir or os.getenv(
            "OFFLINE_RAG_SOURCE_DIR", "data/offline_rag/source",
        ))
        self._manifest_path = self._index_dir / manifest_name
        self._sync_interval_s = sync_interval_s
        self._max_retries = max_retries
        self._retry_backoff_base_s = retry_backoff_base_s

        # State
        self._local_manifest: Dict[str, str] = {}  # chunk_id -> sha256
        self._state = SyncState.IDLE
        self._status = SyncStatus()
        self._bg_task: Optional[asyncio.Task[None]] = None
        self._stop_event: Optional[asyncio.Event] = None  # created lazily in async context
        self._lifecycle_lock = threading.Lock()

        # Load existing manifest
        self._load_manifest()

    # -- Manifest persistence -----------------------------------------------

    def _load_manifest(self) -> None:
        """Load the local sync manifest from disk."""
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path, "r") as f:
                    data = json.load(f)
                self._local_manifest = data.get("chunks", {})
                self._status.local_manifest_chunks = len(self._local_manifest)
                logger.info(
                    "Loaded sync manifest with %d chunks from %s",
                    len(self._local_manifest), self._manifest_path,
                )
            except Exception as exc:
                logger.warning("Failed to load sync manifest: %s", exc)
                self._local_manifest = {}
        else:
            logger.info("No sync manifest found at %s -- starting fresh", self._manifest_path)

    def _save_manifest(self) -> None:
        """Persist the local sync manifest to disk (atomic write with fsync)."""
        self._index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "chunks": self._local_manifest,
        }
        tmp_path = self._manifest_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(self._manifest_path)
            logger.debug("Sync manifest saved (%d chunks)", len(self._local_manifest))
        except (OSError, IOError) as exc:
            logger.error("Failed to save sync manifest: %s", exc)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    # -- Remote manifest ----------------------------------------------------

    def _load_remote_manifest(self) -> Dict[str, str]:
        """Build a manifest from the source directory.

        In production this could be an HTTP fetch; for now we scan the local
        source directory and compute SHA-256 per passage file.
        """
        remote: Dict[str, str] = {}
        passages_file = self._source_dir / "passages.json"

        if passages_file.exists():
            try:
                with open(passages_file, "r") as f:
                    passages = json.load(f)
                for p in passages:
                    chunk_id = p.get("id", "")
                    if not chunk_id:
                        continue
                    text = p.get("text", "")
                    remote[chunk_id] = _sha256_str(text)
            except Exception as exc:
                logger.error("Failed to load remote passages: %s", exc)
        else:
            # Fall back to scanning individual chunk files
            chunks_dir = self._source_dir / "chunks"
            if chunks_dir.is_dir():
                for fp in sorted(chunks_dir.glob("*.json")):
                    try:
                        remote[fp.stem] = _sha256_file(fp)
                    except Exception as exc:
                        logger.warning("Failed to hash chunk %s: %s", fp.name, exc)

        return remote

    # -- Delta computation --------------------------------------------------

    def compute_delta(
        self,
        remote_manifest: Optional[Dict[str, str]] = None,
    ) -> DeltaResult:
        """Compute the delta between local and remote manifests.

        Args:
            remote_manifest: Optional pre-loaded remote manifest.  If ``None``
                the source directory is scanned.

        Returns:
            A ``DeltaResult`` describing additions, updates, and deletions.
        """
        if remote_manifest is None:
            remote_manifest = self._load_remote_manifest()

        local_ids: Set[str] = set(self._local_manifest.keys())
        remote_ids: Set[str] = set(remote_manifest.keys())

        delta = DeltaResult()

        # Additions (in remote but not local)
        for cid in sorted(remote_ids - local_ids):
            delta.additions.append(ChunkDelta(
                chunk_id=cid,
                action="add",
                new_hash=remote_manifest[cid],
            ))

        # Updates (in both, hash differs)
        for cid in sorted(local_ids & remote_ids):
            if self._local_manifest[cid] != remote_manifest[cid]:
                delta.updates.append(ChunkDelta(
                    chunk_id=cid,
                    action="update",
                    old_hash=self._local_manifest[cid],
                    new_hash=remote_manifest[cid],
                ))

        # Deletions (in local but not remote)
        for cid in sorted(local_ids - remote_ids):
            delta.deletions.append(ChunkDelta(
                chunk_id=cid,
                action="delete",
                old_hash=self._local_manifest[cid],
            ))

        logger.info(
            "Delta computed: +%d ~%d -%d",
            len(delta.additions), len(delta.updates), len(delta.deletions),
        )
        return delta

    # -- Apply delta --------------------------------------------------------

    def apply_delta(
        self,
        delta: DeltaResult,
        source_passages: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Apply a computed delta to the local index.

        This copies new/updated passages from source and removes deleted ones.

        Args:
            delta: The delta to apply.
            source_passages: Full list of source passages.  Loaded from the
                source directory if not provided.

        Returns:
            ``True`` if the delta was applied successfully.
        """
        if delta.is_empty:
            logger.info("Delta is empty -- nothing to apply")
            return True

        # Load source passages if needed
        if source_passages is None:
            src_file = self._source_dir / "passages.json"
            if src_file.exists():
                with open(src_file, "r") as f:
                    source_passages = json.load(f)
            else:
                source_passages = []

        source_by_id: Dict[str, Dict[str, Any]] = {
            p.get("id", ""): p for p in source_passages if p.get("id")
        }

        # Load current local passages
        local_passages_path = self._index_dir / "passages.json"
        local_passages: List[Dict[str, Any]] = []
        if local_passages_path.exists():
            try:
                with open(local_passages_path, "r") as f:
                    local_passages = json.load(f)
            except Exception:
                pass

        local_by_id: Dict[str, Dict[str, Any]] = {
            p.get("id", ""): p for p in local_passages if p.get("id")
        }

        try:
            # Apply additions
            for chunk in delta.additions:
                src = source_by_id.get(chunk.chunk_id)
                if src:
                    local_by_id[chunk.chunk_id] = src
                    self._local_manifest[chunk.chunk_id] = chunk.new_hash or ""

            # Apply updates
            for chunk in delta.updates:
                src = source_by_id.get(chunk.chunk_id)
                if src:
                    local_by_id[chunk.chunk_id] = src
                    self._local_manifest[chunk.chunk_id] = chunk.new_hash or ""

            # Apply deletions
            for chunk in delta.deletions:
                local_by_id.pop(chunk.chunk_id, None)
                self._local_manifest.pop(chunk.chunk_id, None)

            # Write updated passages
            self._index_dir.mkdir(parents=True, exist_ok=True)
            updated_passages = list(local_by_id.values())
            tmp = local_passages_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(updated_passages, f, indent=2)
            tmp.replace(local_passages_path)

            # Persist manifest
            self._status.local_manifest_chunks = len(self._local_manifest)
            self._save_manifest()

            logger.info(
                "Delta applied: +%d ~%d -%d => %d total passages",
                len(delta.additions), len(delta.updates),
                len(delta.deletions), len(updated_passages),
            )
            return True

        except Exception as exc:
            logger.error("Failed to apply delta: %s", exc, exc_info=True)
            return False

    # -- Full sync ----------------------------------------------------------

    async def sync(self) -> DeltaResult:
        """Run a full sync cycle: compute delta -> apply -> verify.

        Retries with exponential back-off on failure.
        """
        self._state = SyncState.SYNCING
        self._status.state = SyncState.SYNCING.value
        t0 = time.monotonic()
        last_error: Optional[str] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                remote = self._load_remote_manifest()
                delta = self.compute_delta(remote)

                if delta.is_empty:
                    self._finalise_sync(t0, 0, None)
                    return delta

                success = self.apply_delta(delta)
                if not success:
                    raise RuntimeError("apply_delta returned False")

                # Verify integrity after apply
                if not self._verify_integrity():
                    raise RuntimeError("Post-sync integrity check failed")

                self._finalise_sync(t0, delta.total_changes, None)
                return delta

            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Sync attempt %d/%d failed: %s",
                    attempt, self._max_retries, exc,
                )
                if attempt < self._max_retries:
                    backoff = self._retry_backoff_base_s * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)

        # All retries exhausted
        self._finalise_sync(t0, 0, last_error)
        return DeltaResult()

    def _finalise_sync(
        self, t0: float, changes: int, error: Optional[str],
    ) -> None:
        """Update status after a sync cycle."""
        duration = time.monotonic() - t0
        self._status.last_sync_time = time.time()
        self._status.last_sync_duration_s = round(duration, 3)
        self._status.last_delta_changes = changes
        self._status.total_syncs += 1

        if error:
            self._state = SyncState.ERROR
            self._status.state = SyncState.ERROR.value
            self._status.error_message = error
            self._status.total_errors += 1
            self._status.consecutive_errors += 1
            logger.error("Sync completed with error: %s", error)
        else:
            self._state = SyncState.COMPLETED
            self._status.state = SyncState.COMPLETED.value
            self._status.error_message = None
            self._status.consecutive_errors = 0
            logger.info(
                "Sync completed: %d changes in %.1fs", changes, duration,
            )

    # -- Integrity verification ---------------------------------------------

    def _verify_integrity(self) -> bool:
        """Verify that the local passages match the manifest hashes."""
        passages_path = self._index_dir / "passages.json"
        if not passages_path.exists():
            return len(self._local_manifest) == 0

        try:
            with open(passages_path, "r") as f:
                passages = json.load(f)

            for p in passages:
                cid = p.get("id", "")
                if not cid:
                    continue
                expected = self._local_manifest.get(cid)
                if expected is None:
                    continue
                actual = _sha256_str(p.get("text", ""))
                if actual != expected:
                    logger.warning(
                        "Integrity mismatch for chunk %s: expected %s, got %s",
                        cid, expected[:12], actual[:12],
                    )
                    return False

            return True
        except Exception as exc:
            logger.error("Integrity verification failed: %s", exc)
            return False

    # -- Background sync ----------------------------------------------------

    async def start_background_sync(self) -> None:
        """Start a background loop that syncs at ``_sync_interval_s``.

        Thread-safe: uses a lifecycle lock to prevent double-start.
        The asyncio.Event is created lazily to avoid binding to the wrong loop.
        """
        with self._lifecycle_lock:
            if self._bg_task is not None and not self._bg_task.done():
                logger.warning("Background sync is already running")
                return

            # Create the stop event in the running event loop context
            self._stop_event = asyncio.Event()
            self._stop_event.clear()
            self._bg_task = asyncio.create_task(self._sync_loop())

        logger.info(
            "Background sync started (interval=%.0fs)", self._sync_interval_s,
        )

    async def stop_background_sync(self) -> None:
        """Stop the background sync loop. Thread-safe."""
        with self._lifecycle_lock:
            if self._stop_event is not None:
                self._stop_event.set()
            if self._bg_task is not None:
                self._bg_task.cancel()
                try:
                    await self._bg_task
                except asyncio.CancelledError:
                    pass
                self._bg_task = None
        logger.info("Background sync stopped")

    async def _sync_loop(self) -> None:
        """Internal loop: sync, sleep, repeat."""
        while not self._stop_event.is_set():
            try:
                await self.sync()
            except Exception as exc:
                logger.error("Background sync error: %s", exc, exc_info=True)

            self._status.next_sync_time = time.time() + self._sync_interval_s
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._sync_interval_s,
                )
                break  # stop_event was set
            except asyncio.TimeoutError:
                pass  # interval elapsed, loop again

    # -- P2P sync stub (future) ---------------------------------------------

    async def sync_p2p(self, peer_address: str) -> Dict[str, Any]:
        """Stub for future peer-to-peer sync.

        In the planned mesh topology, edge devices can sync directly with
        each other rather than requiring a central server.

        Args:
            peer_address: Network address of the peer node.

        Returns:
            Status dict from the peer sync attempt.
        """
        logger.info("P2P sync stub called for peer %s -- not yet implemented", peer_address)
        return {
            "status": "not_implemented",
            "peer": peer_address,
            "message": "P2P sync is planned for a future release",
        }

    # -- Status -------------------------------------------------------------

    def get_sync_status(self) -> Dict[str, Any]:
        """Return the current sync engine status as a serialisable dict."""
        return {
            "state": self._status.state,
            "last_sync_time": self._status.last_sync_time,
            "last_sync_duration_s": self._status.last_sync_duration_s,
            "last_delta_changes": self._status.last_delta_changes,
            "total_syncs": self._status.total_syncs,
            "total_errors": self._status.total_errors,
            "consecutive_errors": self._status.consecutive_errors,
            "local_manifest_chunks": self._status.local_manifest_chunks,
            "next_sync_time": self._status.next_sync_time,
            "error_message": self._status.error_message,
            "index_dir": str(self._index_dir),
            "source_dir": str(self._source_dir),
            "sync_interval_s": self._sync_interval_s,
            "background_running": (
                self._bg_task is not None and not self._bg_task.done()
            ),
        }
