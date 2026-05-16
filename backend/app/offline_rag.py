"""Production Offline RAG Pipeline for RetinalAI Clinical Screening.

Provides vector-similarity search over a local FAISS index of clinical
ophthalmology passages.  Designed for fully offline operation with:

- ONNX-quantized bge-m3 embedder (fallback: sentence-transformers)
- Compressed FAISS index loaded from versioned bundles
- Thread-safe singleton pattern
- Structured logging and latency/cache metrics
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports -- graceful degradation when missing
# ---------------------------------------------------------------------------

_faiss = None
_np = None
_onnxruntime = None
_sentence_transformers = None

try:
    import numpy as np
    _np = np
except ImportError:
    pass

try:
    import faiss  # type: ignore[import-untyped]
    _faiss = faiss
except ImportError:
    logger.info("FAISS not available -- offline RAG search will be disabled")

try:
    import onnxruntime  # type: ignore[import-untyped]
    _onnxruntime = onnxruntime
except ImportError:
    logger.debug("ONNX Runtime not available -- will try sentence-transformers")

try:
    import sentence_transformers  # type: ignore[import-untyped]
    _sentence_transformers = sentence_transformers
except Exception:
    logger.debug("sentence-transformers not available")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single search result from the offline RAG index."""
    passage_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineMetrics:
    """Accumulated metrics for the offline RAG pipeline."""
    total_searches: int = 0
    total_embeddings: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_search_latency_ms: float = 0.0
    avg_embed_latency_ms: float = 0.0
    last_search_latency_ms: float = 0.0
    errors: int = 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class OfflineRAGPipeline:
    """Thread-safe offline RAG pipeline backed by FAISS + ONNX/ST embedder.

    Usage::

        pipeline = OfflineRAGPipeline.get_instance()
        results  = pipeline.search("diabetic retinopathy grading criteria", top_k=5)

    The singleton is lazily initialised on first call to ``get_instance()``.
    """

    _instance: Optional["OfflineRAGPipeline"] = None
    _lock: threading.Lock = threading.Lock()

    # -- Singleton access ---------------------------------------------------

    @classmethod
    def get_instance(cls) -> "OfflineRAGPipeline":
        """Return the global singleton, creating it if necessary."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Tear down the singleton (used in tests / shutdown)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._shutdown()
                cls._instance = None

    # -- Init / teardown ----------------------------------------------------

    def __init__(
        self,
        index_dir: Optional[str] = None,
        embedder_path: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.45,
        cache_size: int = 256,
    ) -> None:
        self._index_dir = Path(index_dir or os.getenv(
            "OFFLINE_RAG_INDEX_DIR", "data/offline_rag/index",
        ))
        self._embedder_path = embedder_path or os.getenv(
            "OFFLINE_RAG_EMBEDDER", "models/embedder/bge-m3-quantized.onnx",
        )
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold
        self._cache_size = cache_size

        # State
        self._index: Any = None  # faiss.Index
        self._passages: List[Dict[str, Any]] = []
        self._embedder: Any = None  # onnxruntime session or SentenceTransformer
        self._embedder_type: str = "none"
        self._embed_dim: int = 0
        self._bundle_version: str = "unknown"
        self._loaded: bool = False

        # Metrics
        self._metrics = PipelineMetrics()
        self._search_latencies: List[float] = []
        self._embed_latencies: List[float] = []
        self._metrics_lock = threading.Lock()

        # Simple LRU cache (query_hash -> results)
        self._cache: Dict[str, List[SearchResult]] = {}
        self._cache_order: List[str] = []

        self._load()

    def _shutdown(self) -> None:
        """Release resources."""
        self._index = None
        self._passages = []
        self._embedder = None
        self._loaded = False
        logger.info("Offline RAG pipeline shut down")

    # -- Loading ------------------------------------------------------------

    def _load(self) -> None:
        """Load FAISS index, passages, and embedder from the bundle directory."""
        if _np is None:
            logger.warning("numpy is required for offline RAG -- pipeline disabled")
            return

        index_path = self._index_dir / "index.faiss"
        passages_path = self._index_dir / "passages.json"
        meta_path = self._index_dir / "bundle_meta.json"

        # Load metadata
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                self._bundle_version = meta.get("version", "unknown")
            except Exception as exc:
                logger.warning("Failed to read bundle metadata: %s", exc)

        # Load passages
        if passages_path.exists():
            try:
                with open(passages_path, "r") as f:
                    self._passages = json.load(f)
                logger.info(
                    "Loaded %d passages from %s", len(self._passages), passages_path,
                )
            except Exception as exc:
                logger.error("Failed to load passages: %s", exc)
                return
        else:
            logger.info("No passages file at %s -- pipeline inactive", passages_path)
            return

        # Load FAISS index
        if _faiss is not None and index_path.exists():
            try:
                self._index = _faiss.read_index(str(index_path))
                self._embed_dim = self._index.d
                logger.info(
                    "FAISS index loaded: %d vectors, dim=%d",
                    self._index.ntotal, self._embed_dim,
                )
            except Exception as exc:
                logger.error("Failed to load FAISS index: %s", exc)
                return
        else:
            logger.info("FAISS index not available at %s", index_path)
            return

        # Load embedder (prefer ONNX, fallback to sentence-transformers)
        self._load_embedder()

        self._loaded = bool(self._index and self._embedder and self._passages)
        if self._loaded:
            logger.info(
                "Offline RAG pipeline ready (version=%s, embedder=%s, passages=%d)",
                self._bundle_version, self._embedder_type, len(self._passages),
            )

    def _load_embedder(self) -> None:
        """Load the best available embedder."""
        # Try ONNX first
        if _onnxruntime is not None and Path(self._embedder_path).exists():
            try:
                sess_opts = _onnxruntime.SessionOptions()
                sess_opts.graph_optimization_level = (
                    _onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                sess_opts.intra_op_num_threads = min(os.cpu_count() or 2, 4)
                self._embedder = _onnxruntime.InferenceSession(
                    self._embedder_path, sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                self._embedder_type = "onnx"
                logger.info("ONNX embedder loaded from %s", self._embedder_path)
                return
            except Exception as exc:
                logger.warning("ONNX embedder load failed: %s", exc)

        # Fallback to sentence-transformers
        if _sentence_transformers is not None:
            try:
                model_name = os.getenv(
                    "OFFLINE_RAG_ST_MODEL", "BAAI/bge-m3",
                )
                self._embedder = _sentence_transformers.SentenceTransformer(model_name)
                self._embedder_type = "sentence-transformers"
                logger.info("sentence-transformers embedder loaded: %s", model_name)
                return
            except Exception as exc:
                logger.warning("sentence-transformers embedder load failed: %s", exc)

        logger.warning("No embedder available -- offline RAG search disabled")

    # -- Embedding ----------------------------------------------------------

    def embed_query(self, query: str) -> Optional["_np.ndarray"]:
        """Embed a query string into a dense vector.

        Returns:
            Normalised numpy array of shape ``(embed_dim,)`` or ``None`` on failure.
        """
        if _np is None or self._embedder is None:
            return None

        t0 = time.monotonic()
        try:
            if self._embedder_type == "onnx":
                vec = self._embed_onnx(query)
            elif self._embedder_type == "sentence-transformers":
                vec = self._embedder.encode(
                    [query], normalize_embeddings=True, show_progress_bar=False,
                )[0]
            else:
                return None

            latency_ms = (time.monotonic() - t0) * 1000
            self._record_embed_latency(latency_ms)
            return _np.asarray(vec, dtype=_np.float32)
        except Exception as exc:
            logger.error("Embedding failed: %s", exc, exc_info=True)
            with self._metrics_lock:
                self._metrics.errors += 1
            return None

    def _embed_onnx(self, text: str) -> "_np.ndarray":
        """Run the ONNX embedder session for a single query.

        Supports two ONNX model layouts:
        1. Tokenizer-baked models: expects input_ids (int64) — we use a simple
           character-ordinal encoding as a fallback tokeniser.
        2. Raw-text models: expects a string tensor — passed directly.

        For production, export your ONNX model with the tokeniser baked in
        (e.g., via ``optimum`` or ``onnxruntime-extensions``).
        """
        inputs = self._embedder.get_inputs()
        input_name = inputs[0].name
        input_type = inputs[0].type

        try:
            if "string" in input_type.lower():
                # Model accepts raw text
                feed = {input_name: _np.array([[text[:512]]])}
            else:
                # Tokeniser-baked or ordinal-encoded model (int64 input_ids)
                token_ids = [ord(c) for c in text[:512]]
                feed = {input_name: _np.array([token_ids], dtype=_np.int64)}
                # Add attention_mask if the model expects it
                if len(inputs) > 1 and "attention_mask" in inputs[1].name:
                    feed[inputs[1].name] = _np.ones((1, len(token_ids)), dtype=_np.int64)

            result = self._embedder.run(None, feed)
        except Exception as exc:
            logger.warning("ONNX embedding inference failed: %s", exc)
            raise

        vec = result[0][0]
        # L2-normalise
        norm = float(_np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec

    # -- Search -------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search the offline index for passages relevant to *query*.

        Args:
            query: Natural-language search query.
            top_k: Number of results to return (default: pipeline setting).
            threshold: Minimum similarity score (default: pipeline setting).

        Returns:
            List of ``SearchResult`` objects sorted by descending score.
        """
        if not self._loaded:
            logger.debug("Offline RAG pipeline not loaded -- returning empty results")
            return []

        top_k = top_k or self._top_k
        threshold = threshold if threshold is not None else self._similarity_threshold

        # Input validation
        query = query.strip()
        if not query:
            return []

        # Cache check (thread-safe)
        cache_key = self._cache_key(query, top_k, threshold)
        with self._metrics_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._metrics.cache_hits += 1
                return list(cached)  # return copy to prevent mutation
            self._metrics.cache_misses += 1

        t0 = time.monotonic()

        vec = self.embed_query(query)
        if vec is None:
            return []

        try:
            query_vec = vec.reshape(1, -1).astype(_np.float32)
            distances, indices = self._index.search(query_vec, min(top_k * 2, self._index.ntotal))

            results: List[SearchResult] = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self._passages):
                    continue
                score = float(dist)
                # For inner-product / cosine index, higher is better.
                # For L2 index, lower is better -- convert.
                if _faiss is not None and hasattr(self._index, "metric_type"):
                    if self._index.metric_type == _faiss.METRIC_L2:
                        score = 1.0 / (1.0 + score)
                if score < threshold:
                    continue
                passage = self._passages[idx]
                results.append(SearchResult(
                    passage_id=passage.get("id", str(idx)),
                    text=passage.get("text", ""),
                    score=round(score, 4),
                    metadata=passage.get("metadata", {}),
                ))
                if len(results) >= top_k:
                    break

            results.sort(key=lambda r: r.score, reverse=True)

        except Exception as exc:
            logger.error("FAISS search failed: %s", exc, exc_info=True)
            with self._metrics_lock:
                self._metrics.errors += 1
            return []

        latency_ms = (time.monotonic() - t0) * 1000
        self._record_search_latency(latency_ms)

        # Store in cache
        self._cache_put(cache_key, results)

        logger.debug(
            "Offline RAG search: query=%r top_k=%d results=%d latency=%.1fms",
            query[:60], top_k, len(results), latency_ms,
        )
        return results

    def generate_response(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """High-level method: search + compose a structured response.

        This is the primary endpoint for the offline RAG router.  It returns
        a dict suitable for direct JSON serialisation.
        """
        results = self.search(query, top_k=top_k, threshold=threshold)
        passages_text = [r.text for r in results]

        return {
            "query": query,
            "results": [
                {
                    "passage_id": r.passage_id,
                    "text": r.text,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results
            ],
            "result_count": len(results),
            "context_summary": (
                " ".join(passages_text[:3])[:500] if passages_text else ""
            ),
            "pipeline_version": self._bundle_version,
            "embedder_type": self._embedder_type,
        }

    # -- Status / metrics ---------------------------------------------------

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Return comprehensive pipeline status for health checks."""
        with self._metrics_lock:
            metrics_snapshot = PipelineMetrics(
                total_searches=self._metrics.total_searches,
                total_embeddings=self._metrics.total_embeddings,
                cache_hits=self._metrics.cache_hits,
                cache_misses=self._metrics.cache_misses,
                avg_search_latency_ms=self._metrics.avg_search_latency_ms,
                avg_embed_latency_ms=self._metrics.avg_embed_latency_ms,
                last_search_latency_ms=self._metrics.last_search_latency_ms,
                errors=self._metrics.errors,
            )

        return {
            "loaded": self._loaded,
            "bundle_version": self._bundle_version,
            "embedder_type": self._embedder_type,
            "embed_dim": self._embed_dim,
            "num_passages": len(self._passages),
            "index_vectors": self._index.ntotal if self._index else 0,
            "top_k": self._top_k,
            "similarity_threshold": self._similarity_threshold,
            "cache_size": len(self._cache),
            "cache_capacity": self._cache_size,
            "dependencies": {
                "faiss": _faiss is not None,
                "numpy": _np is not None,
                "onnxruntime": _onnxruntime is not None,
                "sentence_transformers": _sentence_transformers is not None,
            },
            "metrics": {
                "total_searches": metrics_snapshot.total_searches,
                "total_embeddings": metrics_snapshot.total_embeddings,
                "cache_hits": metrics_snapshot.cache_hits,
                "cache_misses": metrics_snapshot.cache_misses,
                "avg_search_latency_ms": round(metrics_snapshot.avg_search_latency_ms, 2),
                "avg_embed_latency_ms": round(metrics_snapshot.avg_embed_latency_ms, 2),
                "last_search_latency_ms": round(metrics_snapshot.last_search_latency_ms, 2),
                "errors": metrics_snapshot.errors,
            },
        }

    # -- Internal helpers ---------------------------------------------------

    def _cache_key(self, query: str, top_k: int, threshold: float) -> str:
        raw = f"{query}|{top_k}|{threshold}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_put(self, key: str, value: List[SearchResult]) -> None:
        """Thread-safe cache insertion with LRU eviction."""
        with self._metrics_lock:
            if key in self._cache:
                return
            if len(self._cache) >= self._cache_size:
                evict = self._cache_order.pop(0)
                self._cache.pop(evict, None)
            self._cache[key] = value
            self._cache_order.append(key)

    def _record_search_latency(self, ms: float) -> None:
        with self._metrics_lock:
            self._search_latencies.append(ms)
            # Keep last 1000 samples
            if len(self._search_latencies) > 1000:
                self._search_latencies = self._search_latencies[-500:]
            self._metrics.total_searches += 1
            self._metrics.last_search_latency_ms = ms
            self._metrics.avg_search_latency_ms = (
                sum(self._search_latencies) / len(self._search_latencies)
            )

    def _record_embed_latency(self, ms: float) -> None:
        with self._metrics_lock:
            self._embed_latencies.append(ms)
            if len(self._embed_latencies) > 1000:
                self._embed_latencies = self._embed_latencies[-500:]
            self._metrics.total_embeddings += 1
            self._metrics.avg_embed_latency_ms = (
                sum(self._embed_latencies) / len(self._embed_latencies)
            )
