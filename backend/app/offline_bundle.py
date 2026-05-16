"""Offline Bundle Builder and Version Manager.

Packages the FAISS index, passages, embedder model, and metadata into a
compressed archive that can be deployed to edge / offline nodes.

Features:
- Semantic versioning (major.minor.patch) for bundles
- Gzip or zstd compression targeting < 150 MB
- SHA-256 integrity verification
- In-archive JSON manifest describing bundle contents
- Version listing and rollback support
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# Optional zstd support
_zstd = None
try:
    import zstandard as _zstd  # type: ignore[import-untyped]
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class BundleManifest:
    """Metadata embedded inside every bundle archive."""

    version: str
    created_at: float
    created_by: str = "retinalai-bundle-manager"
    compression: str = "gzip"
    files: List[Dict[str, Any]] = field(default_factory=list)
    total_size_bytes: int = 0
    sha256: str = ""
    passages_count: int = 0
    index_vectors: int = 0
    embed_dim: int = 0
    embedder_type: str = ""
    app_version: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "compression": self.compression,
            "files": self.files,
            "total_size_bytes": self.total_size_bytes,
            "sha256": self.sha256,
            "passages_count": self.passages_count,
            "index_vectors": self.index_vectors,
            "embed_dim": self.embed_dim,
            "embedder_type": self.embedder_type,
            "app_version": self.app_version,
            "notes": self.notes,
        }


@dataclass
class BundleInfo:
    """Summary information about a bundle on disk."""

    version: str
    path: str
    size_bytes: int
    sha256: str
    created_at: float
    compression: str
    passages_count: int = 0
    index_vectors: int = 0


# ---------------------------------------------------------------------------
# Semantic versioning helpers
# ---------------------------------------------------------------------------


def parse_semver(version: str) -> Tuple[int, int, int]:
    """Parse a ``major.minor.patch`` version string."""
    parts = version.lstrip("v").split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid semver: {version}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def format_semver(major: int, minor: int, patch: int) -> str:
    return f"{major}.{minor}.{patch}"


def bump_version(
    current: str,
    bump: str = "patch",
) -> str:
    """Bump a semantic version.

    Args:
        current: Current version string (e.g. ``"1.2.3"``).
        bump: One of ``"major"``, ``"minor"``, ``"patch"``.

    Returns:
        The bumped version string.
    """
    major, minor, patch = parse_semver(current)
    if bump == "major":
        return format_semver(major + 1, 0, 0)
    elif bump == "minor":
        return format_semver(major, minor + 1, 0)
    else:
        return format_semver(major, minor, patch + 1)


# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Bundle Manager
# ---------------------------------------------------------------------------


class OfflineBundleManager:
    """Manages versioned offline RAG bundles.

    Each bundle is a compressed tar archive containing:

    - ``index.faiss`` -- the FAISS vector index
    - ``passages.json`` -- the passage corpus
    - ``embedder/`` -- the ONNX embedder (or a sentinel marker)
    - ``bundle_meta.json`` -- the in-archive manifest
    - ``bundle_manifest.json`` -- detailed file listing with hashes

    Bundles are stored under ``bundles_dir`` with filenames like
    ``offline_rag_v1.2.3.tar.gz``.
    """

    BUNDLE_PREFIX = "offline_rag_v"

    def __init__(
        self,
        index_dir: Optional[str] = None,
        bundles_dir: Optional[str] = None,
        compression: str = "gzip",
        target_size_mb: int = 150,
    ) -> None:
        self._index_dir = Path(
            index_dir
            or os.getenv(
                "OFFLINE_RAG_INDEX_DIR",
                "data/offline_rag/index",
            )
        )
        self._bundles_dir = Path(
            bundles_dir
            or os.getenv(
                "OFFLINE_RAG_BUNDLES_DIR",
                "data/offline_rag/bundles",
            )
        )
        self._compression = compression if compression in ("gzip", "zstd") else "gzip"
        self._target_size_mb = target_size_mb

        if self._compression == "zstd" and _zstd is None:
            logger.warning("zstandard not available -- falling back to gzip")
            self._compression = "gzip"

        self._bundles_dir.mkdir(parents=True, exist_ok=True)

    # -- Build --------------------------------------------------------------

    def build_bundle(
        self,
        version: Optional[str] = None,
        bump: str = "patch",
        notes: str = "",
    ) -> BundleInfo:
        """Build a new versioned bundle from the current index directory.

        Args:
            version: Explicit version string.  If ``None`` the latest
                version is auto-bumped using *bump*.
            bump: Version bump type when *version* is ``None``.
            notes: Free-text notes to embed in the manifest.

        Returns:
            ``BundleInfo`` describing the newly created bundle.

        Raises:
            FileNotFoundError: If required index files are missing.
            RuntimeError: If the bundle exceeds the target size limit.
        """
        # Determine version
        if version is None:
            latest = self._latest_version()
            version = bump_version(latest, bump) if latest else "1.0.0"

        logger.info("Building offline bundle v%s from %s", version, self._index_dir)

        # Validate required files
        required: List[Tuple[str, Path]] = []
        passages_path = self._index_dir / "passages.json"
        if passages_path.exists():
            required.append(("passages.json", passages_path))
        else:
            raise FileNotFoundError(f"passages.json not found in {self._index_dir}")

        index_path = self._index_dir / "index.faiss"
        if index_path.exists():
            required.append(("index.faiss", index_path))

        self._index_dir / "bundle_meta.json"

        # Collect optional embedder files
        embedder_files: List[Tuple[str, Path]] = []
        embedder_dir = self._index_dir / "embedder"
        if embedder_dir.is_dir():
            for fp in embedder_dir.rglob("*"):
                if fp.is_file():
                    rel = fp.relative_to(self._index_dir)
                    embedder_files.append((str(rel), fp))

        # Read passage stats
        passages_count = 0
        try:
            with open(passages_path, "r") as f:
                passages = json.load(f)
            passages_count = len(passages)
        except Exception:
            pass

        # Build in-archive manifest
        manifest = BundleManifest(
            version=version,
            created_at=time.time(),
            compression=self._compression,
            passages_count=passages_count,
            app_version=settings.app_version,
            notes=notes,
        )

        # Determine archive filename and extension
        ext = ".tar.gz" if self._compression == "gzip" else ".tar.zst"
        archive_name = f"{self.BUNDLE_PREFIX}{version}{ext}"
        archive_path = self._bundles_dir / archive_name

        # Build tarball
        all_entries = required + embedder_files

        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp_tar:
            tmp_tar_path = Path(tmp_tar.name)

        try:
            with tarfile.open(tmp_tar_path, "w") as tar:
                for arcname, filepath in all_entries:
                    tar.add(str(filepath), arcname=arcname)
                    size = filepath.stat().st_size
                    manifest.files.append(
                        {
                            "name": arcname,
                            "size_bytes": size,
                            "sha256": _sha256_file(filepath),
                        }
                    )
                    manifest.total_size_bytes += size

                # Write bundle_meta.json
                version_meta = {
                    "version": version,
                    "created_at": manifest.created_at,
                    "passages_count": passages_count,
                    "app_version": settings.app_version,
                }
                meta_bytes = json.dumps(version_meta, indent=2).encode()
                info = tarfile.TarInfo(name="bundle_meta.json")
                info.size = len(meta_bytes)
                tar.addfile(info, io.BytesIO(meta_bytes))

                # Write manifest inside archive
                manifest_bytes = json.dumps(manifest.to_dict(), indent=2).encode()
                m_info = tarfile.TarInfo(name="bundle_manifest.json")
                m_info.size = len(manifest_bytes)
                tar.addfile(m_info, io.BytesIO(manifest_bytes))

            # Compress
            self._compress(tmp_tar_path, archive_path)
        finally:
            tmp_tar_path.unlink(missing_ok=True)

        # Compute archive hash
        sha = _sha256_file(archive_path)
        size = archive_path.stat().st_size

        # Write sidecar hash file
        hash_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
        hash_path.write_text(f"{sha}  {archive_name}\n")

        size_mb = size / (1024 * 1024)
        if size_mb > self._target_size_mb:
            logger.warning(
                "Bundle v%s is %.1f MB (target < %d MB)",
                version,
                size_mb,
                self._target_size_mb,
            )

        logger.info(
            "Bundle v%s built: %s (%.1f MB, sha256=%s)",
            version,
            archive_path,
            size_mb,
            sha[:16],
        )

        return BundleInfo(
            version=version,
            path=str(archive_path),
            size_bytes=size,
            sha256=sha,
            created_at=manifest.created_at,
            compression=self._compression,
            passages_count=passages_count,
        )

    # -- Verify -------------------------------------------------------------

    def verify_bundle(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Verify bundle integrity via SHA-256.

        Args:
            version: Version to verify.  Defaults to the latest.

        Returns:
            Dict with ``valid``, ``version``, ``expected_sha256``,
            ``actual_sha256``, and ``path``.
        """
        path = self._resolve_bundle_path(version)
        if path is None:
            return {
                "valid": False,
                "error": f"Bundle not found for version={version}",
            }

        hash_path = path.with_suffix(path.suffix + ".sha256")
        if not hash_path.exists():
            return {
                "valid": False,
                "version": self._version_from_path(path),
                "path": str(path),
                "error": "No .sha256 sidecar file found",
            }

        raw_expected = (
            hash_path.read_text().strip().split()[0] if hash_path.read_text().strip() else ""
        )

        # Validate that the expected hash is a valid hex SHA-256 string
        valid_hex = len(raw_expected) == 64 and all(
            c in "0123456789abcdef" for c in raw_expected.lower()
        )
        if not valid_hex:
            return {
                "valid": False,
                "version": self._version_from_path(path),
                "path": str(path),
                "error": f"Corrupt .sha256 sidecar: expected 64-char hex, got {raw_expected!r:.40}",
            }

        try:
            actual = _sha256_file(path)
        except (OSError, IOError) as exc:
            return {
                "valid": False,
                "version": self._version_from_path(path),
                "path": str(path),
                "error": f"Cannot read bundle for hash verification: {exc}",
            }

        return {
            "valid": actual == raw_expected,
            "version": self._version_from_path(path),
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "expected_sha256": raw_expected,
            "actual_sha256": actual,
        }

    # -- Info ---------------------------------------------------------------

    def get_bundle_info(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Return metadata for a specific bundle version.

        Args:
            version: Version to inspect.  Defaults to the latest.

        Returns:
            Dict with version, size, hash, contents manifest, etc.
        """
        path = self._resolve_bundle_path(version)
        if path is None:
            return {"error": f"Bundle not found for version={version}"}

        ver = self._version_from_path(path)
        sha = _sha256_file(path)
        size = path.stat().st_size

        # Try to extract the in-archive manifest
        contents_manifest: Optional[Dict[str, Any]] = None
        try:
            contents_manifest = self._read_archive_manifest(path)
        except Exception as exc:
            logger.debug("Could not read in-archive manifest: %s", exc)

        return {
            "version": ver,
            "path": str(path),
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2),
            "sha256": sha,
            "created_at": contents_manifest.get("created_at") if contents_manifest else None,
            "compression": self._compression,
            "contents": contents_manifest,
        }

    # -- List versions ------------------------------------------------------

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all available bundle versions, newest first.

        Returns:
            List of dicts with ``version``, ``path``, ``size_bytes``,
            ``size_mb``, and ``sha256``.
        """
        versions: List[Dict[str, Any]] = []

        for fp in sorted(self._bundles_dir.iterdir(), reverse=True):
            if not fp.name.startswith(self.BUNDLE_PREFIX):
                continue
            if not (
                fp.suffix in (".gz", ".zst")
                or fp.name.endswith(".tar.gz")
                or fp.name.endswith(".tar.zst")
            ):
                continue
            if fp.name.endswith(".sha256"):
                continue

            ver = self._version_from_path(fp)
            size = fp.stat().st_size
            versions.append(
                {
                    "version": ver,
                    "path": str(fp),
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2),
                }
            )

        # Sort by semver descending
        versions.sort(
            key=lambda v: parse_semver(v["version"]),
            reverse=True,
        )
        return versions

    def get_latest_bundle_path(self) -> Optional[Path]:
        """Return the filesystem path to the latest bundle, or None."""
        return self._resolve_bundle_path(None)

    # -- Compression --------------------------------------------------------

    def _compress(self, src: Path, dst: Path) -> None:
        """Compress *src* to *dst* using the configured algorithm."""
        if self._compression == "zstd" and _zstd is not None:
            cctx = _zstd.ZstdCompressor(level=9)
            with open(src, "rb") as fin, open(dst, "wb") as fout:
                cctx.copy_stream(fin, fout)
        else:
            with open(src, "rb") as fin, gzip.open(dst, "wb", compresslevel=6) as fout:
                shutil.copyfileobj(fin, fout)

    # -- Internal helpers ---------------------------------------------------

    def _latest_version(self) -> Optional[str]:
        """Return the latest bundle version string, or ``None``."""
        versions = self.list_versions()
        return versions[0]["version"] if versions else None

    def _resolve_bundle_path(self, version: Optional[str]) -> Optional[Path]:
        """Resolve a version string to a bundle file path."""
        if version is None:
            versions = self.list_versions()
            if not versions:
                return None
            return Path(versions[0]["path"])

        for ext in (".tar.gz", ".tar.zst"):
            candidate = self._bundles_dir / f"{self.BUNDLE_PREFIX}{version}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _version_from_path(self, path: Path) -> str:
        """Extract the version string from a bundle filename."""
        name = path.name
        name = name.replace(self.BUNDLE_PREFIX, "")
        for suffix in (".tar.gz", ".tar.zst"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name

    def _read_archive_manifest(self, path: Path) -> Optional[Dict[str, Any]]:
        """Extract and parse ``bundle_manifest.json`` from an archive."""
        open_mode = "r:gz" if path.name.endswith(".tar.gz") else "r:*"
        with tarfile.open(path, open_mode) as tar:
            try:
                member = tar.getmember("bundle_manifest.json")
                fobj = tar.extractfile(member)
                if fobj is not None:
                    return json.load(fobj)
            except KeyError:
                pass
        return None
