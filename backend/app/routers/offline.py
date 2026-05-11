"""Offline RAG API router.

Exposes the offline RAG pipeline, delta sync engine, and bundle manager
via a versioned REST API.  All endpoints are gated behind the
``OFFLINE_RAG__ENABLED`` feature flag.

Endpoints
---------
GET  /v1/offline/status       -- pipeline + sync + bundle health
POST /v1/offline/sync         -- trigger a delta sync (background task)
GET  /v1/offline/bundle       -- download the current bundle archive
GET  /v1/offline/bundle/info  -- bundle metadata (version, size, hash)
POST /v1/offline/search       -- offline RAG search
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/offline", tags=["offline-rag"])

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

# The offline_rag settings block may not exist on Settings yet (it is
# configured via the OFFLINE_RAG__ENABLED env var).  We check at request
# time via a helper so the app can start regardless.

_ENABLED: Optional[bool] = None


def _is_enabled() -> bool:
    """Check if offline RAG is enabled via settings or env var."""
    global _ENABLED
    if _ENABLED is None:
        # Prefer the pydantic settings value; fall back to raw env var
        _ENABLED = getattr(settings, "offline_rag", None) is not None and settings.offline_rag.enabled
        if not _ENABLED:
            _ENABLED = os.getenv("OFFLINE_RAG__ENABLED", "false").lower() in (
                "1", "true", "yes",
            )
    return _ENABLED


def _require_enabled() -> None:
    """Raise 404 if the feature flag is off."""
    if not _is_enabled():
        raise HTTPException(
            status_code=404,
            detail="Offline RAG is not enabled. Set OFFLINE_RAG__ENABLED=true.",
        )


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_pipeline = None
_sync_engine = None
_bundle_manager = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from backend.app.offline_rag import OfflineRAGPipeline
        _pipeline = OfflineRAGPipeline.get_instance()
    return _pipeline


def _get_sync_engine():
    global _sync_engine
    if _sync_engine is None:
        from backend.app.offline_sync import DeltaSyncEngine
        _sync_engine = DeltaSyncEngine()
    return _sync_engine


def _get_bundle_manager():
    global _bundle_manager
    if _bundle_manager is None:
        from backend.app.offline_bundle import OfflineBundleManager
        _bundle_manager = OfflineBundleManager()
    return _bundle_manager


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Request body for offline RAG search."""
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    threshold: float = Field(
        0.45, ge=0.0, le=1.0, description="Minimum similarity threshold",
    )


class SearchResultItem(BaseModel):
    """A single passage result."""
    passage_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    """Response for offline RAG search."""
    query: str
    results: List[SearchResultItem]
    result_count: int
    context_summary: str
    pipeline_version: str
    embedder_type: str


class SyncTriggerRequest(BaseModel):
    """Optional body for sync trigger."""
    force: bool = Field(False, description="Force sync even if recently synced")


class SyncTriggerResponse(BaseModel):
    """Response after triggering a sync."""
    status: str
    message: str


class BundleInfoResponse(BaseModel):
    """Bundle metadata response."""
    version: Optional[str] = None
    path: Optional[str] = None
    size_bytes: Optional[int] = None
    size_mb: Optional[float] = None
    sha256: Optional[str] = None
    created_at: Optional[float] = None
    compression: Optional[str] = None
    contents: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class OfflineStatusResponse(BaseModel):
    """Aggregated status for the entire offline subsystem."""
    enabled: bool
    pipeline: Dict[str, Any]
    sync: Dict[str, Any]
    bundle: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status", response_model=OfflineStatusResponse)
async def offline_status() -> Dict[str, Any]:
    """Return offline RAG pipeline status, bundle info, and sync status.

    Always returns a response (even when disabled) so dashboards can
    detect the feature state.
    """
    if not _is_enabled():
        return {
            "enabled": False,
            "pipeline": {"loaded": False, "message": "Offline RAG disabled"},
            "sync": {"state": "disabled"},
            "bundle": {"message": "Offline RAG disabled"},
        }

    pipeline = _get_pipeline()
    sync_engine = _get_sync_engine()
    bundle_mgr = _get_bundle_manager()

    # Bundle: try to get info for the latest version
    bundle_info: Dict[str, Any]
    try:
        latest_path = bundle_mgr.get_latest_bundle_path()
        if latest_path is not None:
            bundle_info = bundle_mgr.get_bundle_info()
        else:
            bundle_info = {"message": "No bundles available"}
    except Exception as exc:
        bundle_info = {"error": str(exc)}

    return {
        "enabled": True,
        "pipeline": pipeline.get_pipeline_status(),
        "sync": sync_engine.get_sync_status(),
        "bundle": bundle_info,
    }


@router.post("/sync", response_model=SyncTriggerResponse)
async def trigger_sync(
    body: Optional[SyncTriggerRequest] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> Dict[str, str]:
    """Trigger a delta sync in the background.

    The sync runs asynchronously; poll ``GET /status`` for progress.
    """
    _require_enabled()

    sync_engine = _get_sync_engine()
    status = sync_engine.get_sync_status()

    if status.get("state") == "syncing" and not (body and body.force):
        return {
            "status": "already_running",
            "message": "A sync is already in progress. Set force=true to queue another.",
        }

    async def _run_sync() -> None:
        try:
            await sync_engine.sync()
        except Exception as exc:
            logger.error("Background sync failed: %s", exc, exc_info=True)

    background_tasks.add_task(_run_sync)
    logger.info("Delta sync triggered (force=%s)", body.force if body else False)

    return {
        "status": "started",
        "message": "Delta sync started in background",
    }


@router.get("/bundle")
async def download_bundle(
    version: Optional[str] = Query(None, description="Bundle version to download"),
) -> StreamingResponse:
    """Download the current (or specified) offline bundle as a streaming archive.

    Returns a ``StreamingResponse`` with ``application/gzip`` or
    ``application/zstd`` content type.
    """
    _require_enabled()

    bundle_mgr = _get_bundle_manager()
    path = bundle_mgr._resolve_bundle_path(version)

    if path is None or not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Bundle not found (version={version or 'latest'})",
        )

    content_type = (
        "application/zstd" if path.name.endswith(".tar.zst")
        else "application/gzip"
    )

    def _iter_file():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 16)  # 64 KB chunks
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        _iter_file(),
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{path.name}"',
            "Content-Length": str(path.stat().st_size),
        },
    )


@router.get("/bundle/info", response_model=BundleInfoResponse)
async def bundle_info(
    version: Optional[str] = Query(None, description="Bundle version to inspect"),
) -> Dict[str, Any]:
    """Return metadata for the latest (or specified) bundle version.

    Includes version, size, SHA-256 hash, compression type, and the
    in-archive contents manifest.
    """
    _require_enabled()

    bundle_mgr = _get_bundle_manager()
    info = bundle_mgr.get_bundle_info(version)
    return info


@router.get("/bundle/versions")
async def bundle_versions() -> Dict[str, Any]:
    """List all available bundle versions."""
    _require_enabled()

    bundle_mgr = _get_bundle_manager()
    versions = bundle_mgr.list_versions()
    return {"versions": versions, "total": len(versions)}


@router.post("/bundle/verify")
async def verify_bundle(
    version: Optional[str] = Query(None, description="Bundle version to verify"),
) -> Dict[str, Any]:
    """Verify bundle integrity via SHA-256 checksum."""
    _require_enabled()

    bundle_mgr = _get_bundle_manager()
    result = bundle_mgr.verify_bundle(version)
    return result


# ---------------------------------------------------------------------------
# Delta sync for mobile bundle updates
# ---------------------------------------------------------------------------


class DeltaRequest(BaseModel):
    """Client sends its current bundle state for delta computation."""
    current_version: str = Field(..., description="Client's current bundle version")
    component_hashes: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of component filename -> SHA-256 hash on client",
    )


class DeltaComponent(BaseModel):
    """A single component that needs updating."""
    filename: str
    action: str = Field(..., description="add | update | delete")
    sha256: str = ""
    size_bytes: int = 0
    download_url: str = ""


class DeltaResponse(BaseModel):
    """Server response with components that changed."""
    current_version: str
    target_version: str
    changed_components: List[DeltaComponent]
    total_download_bytes: int
    is_full_update: bool = False


@router.post("/bundle/delta", response_model=DeltaResponse)
async def compute_bundle_delta(body: DeltaRequest) -> Dict[str, Any]:
    """Compute delta between client's bundle and the latest server bundle.

    The client sends its current version and component hashes. The server
    compares against the latest bundle manifest and returns only changed
    components, minimizing bandwidth for daily sync over 2G/3G.
    """
    _require_enabled()

    bundle_mgr = _get_bundle_manager()

    # Get latest bundle manifest
    try:
        latest_info = bundle_mgr.get_bundle_info()
        if not latest_info or latest_info.get("error"):
            raise HTTPException(status_code=404, detail="No bundles available on server")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Bundle lookup failed: {exc}")

    target_version = latest_info.get("version", "unknown")

    # If the client is already up-to-date, return empty delta
    if body.current_version == target_version:
        return {
            "current_version": body.current_version,
            "target_version": target_version,
            "changed_components": [],
            "total_download_bytes": 0,
            "is_full_update": False,
        }

    # Compute delta by comparing hashes
    server_manifest = latest_info.get("contents", {})
    server_files = server_manifest.get("files", {}) if isinstance(server_manifest, dict) else {}

    changed = []
    total_bytes = 0

    for filename, server_hash in server_files.items():
        client_hash = body.component_hashes.get(filename)
        size = server_manifest.get("file_sizes", {}).get(filename, 0) if isinstance(server_manifest, dict) else 0

        if client_hash is None:
            # New component
            changed.append(DeltaComponent(
                filename=filename,
                action="add",
                sha256=server_hash if isinstance(server_hash, str) else "",
                size_bytes=size,
                download_url=f"/api/v1/offline/bundle/component/{filename}",
            ))
            total_bytes += size
        elif client_hash != server_hash:
            # Updated component
            changed.append(DeltaComponent(
                filename=filename,
                action="update",
                sha256=server_hash if isinstance(server_hash, str) else "",
                size_bytes=size,
                download_url=f"/api/v1/offline/bundle/component/{filename}",
            ))
            total_bytes += size

    # Check for deleted components
    for client_file in body.component_hashes:
        if client_file not in server_files:
            changed.append(DeltaComponent(
                filename=client_file,
                action="delete",
            ))

    is_full = len(changed) > len(server_files) * 0.5

    return {
        "current_version": body.current_version,
        "target_version": target_version,
        "changed_components": [c.model_dump() for c in changed],
        "total_download_bytes": total_bytes,
        "is_full_update": is_full,
    }


@router.post("/search", response_model=SearchResponse)
async def offline_search(body: SearchRequest) -> Dict[str, Any]:
    """Run an offline RAG search query.

    Searches the local FAISS index for passages relevant to the query
    and returns ranked results with similarity scores.
    """
    _require_enabled()

    pipeline = _get_pipeline()

    if not pipeline._loaded:
        raise HTTPException(
            status_code=503,
            detail="Offline RAG pipeline is not loaded. Check /v1/offline/status for details.",
        )

    response = pipeline.generate_response(
        query=body.query,
        top_k=body.top_k,
        threshold=body.threshold,
    )

    logger.info(
        "Offline search: query=%r results=%d",
        body.query[:60], response["result_count"],
    )

    return response
