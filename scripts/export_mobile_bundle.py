#!/usr/bin/env python3
"""Export and validate mobile bundle for Flutter offline-first deployment.

Packages quantized model + FAISS index + ONNX embedder + voice models +
assets into a size-enforced Flutter-compatible bundle.

Usage
-----
    # Build full mobile bundle
    python scripts/export_mobile_bundle.py --output-dir outputs/mobile_bundle

    # Custom size limit
    python scripts/export_mobile_bundle.py --max-bundle-mb 700 --output-dir outputs/mobile_bundle

    # Include voice models
    python scripts/export_mobile_bundle.py --include-voice --output-dir outputs/mobile_bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tarfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_BUNDLE_SIZE_MB = 800  # Hard limit for mobile bundle
COMPONENT_BUDGETS_MB = {
    "quantized_llm": 400,  # Gemma-2-2B Q4_K_M or distilled 3B
    "embedder_onnx": 80,  # bge-m3 4-bit ONNX
    "faiss_index": 80,  # FAISS vector index
    "voice_asr": 75,  # Whisper-tiny
    "voice_tts": 60,  # Piper/Sherpa TTS
    "assets": 30,  # Passage data, configs, metadata
    "flutter_app": 75,  # Flutter app overhead
}


@dataclass
class BundleComponent:
    """A component of the mobile bundle."""

    name: str
    source_path: Optional[str] = None
    size_mb: float = 0.0
    budget_mb: float = 0.0
    included: bool = False
    sha256: str = ""
    compressed_size_mb: float = 0.0


@dataclass
class BundleManifest:
    """Manifest describing the mobile bundle contents."""

    version: str = "1.0.0"
    build_timestamp: str = ""
    total_size_mb: float = 0.0
    compressed_size_mb: float = 0.0
    max_size_mb: float = MAX_BUNDLE_SIZE_MB
    within_budget: bool = False
    components: list[dict] = field(default_factory=list)
    flutter_assets_dir: str = ""
    integrity_sha256: str = ""
    platform: str = "android"
    min_sdk: int = 21
    min_ram_mb: int = 4096


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------


def file_size_mb(path: str | Path) -> float:
    """Get file or directory size in MB."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024 * 1024)
    return 0.0


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_directory(path: str | Path) -> str:
    """Compute SHA-256 hash of all files in a directory."""
    h = hashlib.sha256()
    for fpath in sorted(Path(path).rglob("*")):
        if fpath.is_file():
            h.update(fpath.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Component discovery
# ---------------------------------------------------------------------------


def discover_components(
    include_voice: bool = False,
) -> list[BundleComponent]:
    """Discover available model artifacts for bundling."""
    components: list[BundleComponent] = []

    # Quantized LLM
    llm_candidates = [
        _PROJECT_ROOT / "outputs" / "quantized" / "model_gguf_q4_k_m.bin",
        _PROJECT_ROOT / "outputs" / "quantized" / "model_gguf_q4_k_m.gguf",
        _PROJECT_ROOT / "outputs" / "optimized" / "model_int8_dynamic.pth",
        _PROJECT_ROOT / "models" / "model_vignn_rank1.pth",
    ]
    for candidate in llm_candidates:
        if candidate.exists():
            components.append(
                BundleComponent(
                    name="quantized_llm",
                    source_path=str(candidate),
                    size_mb=file_size_mb(candidate),
                    budget_mb=COMPONENT_BUDGETS_MB["quantized_llm"],
                    included=True,
                    sha256=sha256_file(candidate),
                )
            )
            break
    else:
        components.append(
            BundleComponent(
                name="quantized_llm",
                budget_mb=COMPONENT_BUDGETS_MB["quantized_llm"],
                included=False,
            )
        )

    # ONNX embedder
    embedder_candidates = [
        _PROJECT_ROOT / "models" / "export" / "embedder_bge_m3_int8.onnx",
        _PROJECT_ROOT / "models" / "export" / "model.onnx",
        _PROJECT_ROOT / "outputs" / "quantized" / "model_onnx_optimized.onnx",
    ]
    for candidate in embedder_candidates:
        if candidate.exists():
            components.append(
                BundleComponent(
                    name="embedder_onnx",
                    source_path=str(candidate),
                    size_mb=file_size_mb(candidate),
                    budget_mb=COMPONENT_BUDGETS_MB["embedder_onnx"],
                    included=True,
                    sha256=sha256_file(candidate),
                )
            )
            break
    else:
        components.append(
            BundleComponent(
                name="embedder_onnx",
                budget_mb=COMPONENT_BUDGETS_MB["embedder_onnx"],
                included=False,
            )
        )

    # FAISS index
    faiss_candidates = [
        _PROJECT_ROOT / "data" / "offline_rag" / "index" / "index.faiss",
        _PROJECT_ROOT / "outputs" / "offline_rag" / "index.faiss",
    ]
    for candidate in faiss_candidates:
        if candidate.exists():
            components.append(
                BundleComponent(
                    name="faiss_index",
                    source_path=str(candidate),
                    size_mb=file_size_mb(candidate),
                    budget_mb=COMPONENT_BUDGETS_MB["faiss_index"],
                    included=True,
                    sha256=sha256_file(candidate),
                )
            )
            break
    else:
        components.append(
            BundleComponent(
                name="faiss_index",
                budget_mb=COMPONENT_BUDGETS_MB["faiss_index"],
                included=False,
            )
        )

    # Voice models (optional)
    if include_voice:
        # ASR (Whisper-tiny)
        asr_candidates = [
            _PROJECT_ROOT / "models" / "voice" / "whisper-tiny.onnx",
            _PROJECT_ROOT / "models" / "voice" / "whisper-tiny.bin",
        ]
        for candidate in asr_candidates:
            if candidate.exists():
                components.append(
                    BundleComponent(
                        name="voice_asr",
                        source_path=str(candidate),
                        size_mb=file_size_mb(candidate),
                        budget_mb=COMPONENT_BUDGETS_MB["voice_asr"],
                        included=True,
                        sha256=sha256_file(candidate),
                    )
                )
                break
        else:
            components.append(
                BundleComponent(
                    name="voice_asr",
                    budget_mb=COMPONENT_BUDGETS_MB["voice_asr"],
                    included=False,
                )
            )

        # TTS (Piper/Sherpa)
        tts_candidates = [
            _PROJECT_ROOT / "models" / "voice" / "piper-en-ug.onnx",
            _PROJECT_ROOT / "models" / "voice" / "sherpa-tts.onnx",
        ]
        for candidate in tts_candidates:
            if candidate.exists():
                components.append(
                    BundleComponent(
                        name="voice_tts",
                        source_path=str(candidate),
                        size_mb=file_size_mb(candidate),
                        budget_mb=COMPONENT_BUDGETS_MB["voice_tts"],
                        included=True,
                        sha256=sha256_file(candidate),
                    )
                )
                break
        else:
            components.append(
                BundleComponent(
                    name="voice_tts",
                    budget_mb=COMPONENT_BUDGETS_MB["voice_tts"],
                    included=False,
                )
            )

    # Assets (passages, configs)
    assets_dir = _PROJECT_ROOT / "data" / "offline_rag"
    if assets_dir.exists():
        components.append(
            BundleComponent(
                name="assets",
                source_path=str(assets_dir),
                size_mb=file_size_mb(assets_dir),
                budget_mb=COMPONENT_BUDGETS_MB["assets"],
                included=True,
                sha256=sha256_directory(assets_dir),
            )
        )
    else:
        components.append(
            BundleComponent(
                name="assets",
                budget_mb=COMPONENT_BUDGETS_MB["assets"],
                included=False,
            )
        )

    return components


# ---------------------------------------------------------------------------
# Bundle builder
# ---------------------------------------------------------------------------


def build_bundle(
    components: list[BundleComponent],
    output_dir: Path,
    max_size_mb: float = MAX_BUNDLE_SIZE_MB,
    version: str = "1.0.0",
) -> BundleManifest:
    """Build the mobile bundle archive with size enforcement.

    Parameters
    ----------
    components : list[BundleComponent]
        Discovered bundle components.
    output_dir : Path
        Output directory for the bundle.
    max_size_mb : float
        Maximum allowed total bundle size in MB.
    version : str
        Semantic version for the bundle.

    Returns
    -------
    BundleManifest
        Manifest describing the built bundle.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / "staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    # Flutter asset structure
    flutter_assets = staging_dir / "flutter_assets" / "models"
    flutter_assets.mkdir(parents=True)

    included_components = [c for c in components if c.included and c.source_path]
    total_size = 0.0

    for comp in included_components:
        src = Path(comp.source_path)
        if comp.size_mb > comp.budget_mb:
            logger.warning(
                "Component %s (%.1f MB) exceeds budget (%.1f MB) — including anyway",
                comp.name,
                comp.size_mb,
                comp.budget_mb,
            )

        dest = flutter_assets / comp.name
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Use the original file extension
            dest_file = flutter_assets / f"{comp.name}{src.suffix}"
            shutil.copy2(src, dest_file)

        total_size += comp.size_mb
        logger.info("Staged: %s (%.1f MB)", comp.name, comp.size_mb)

    # Write component manifest inside bundle
    manifest_data = {
        "version": version,
        "components": [asdict(c) for c in components],
        "total_uncompressed_mb": round(total_size, 2),
    }
    manifest_path = staging_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2))

    # Compress to tar.gz
    archive_name = f"retinalai_mobile_v{version}.tar.gz"
    archive_path = output_dir / archive_name
    logger.info("Compressing bundle to %s ...", archive_path)

    with tarfile.open(archive_path, "w:gz", compresslevel=9) as tar:
        tar.add(staging_dir, arcname="retinalai_mobile")

    compressed_size_mb = file_size_mb(archive_path)
    bundle_sha256 = sha256_file(archive_path)

    # Write SHA-256 sidecar
    sha_path = output_dir / f"{archive_name}.sha256"
    sha_path.write_text(f"{bundle_sha256}  {archive_name}\n")

    # Check budget
    within_budget = total_size <= max_size_mb

    # Cleanup staging
    shutil.rmtree(staging_dir)

    manifest = BundleManifest(
        version=version,
        build_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        total_size_mb=round(total_size, 2),
        compressed_size_mb=round(compressed_size_mb, 2),
        max_size_mb=max_size_mb,
        within_budget=within_budget,
        components=[asdict(c) for c in components],
        flutter_assets_dir=str(output_dir / "flutter_assets"),
        integrity_sha256=bundle_sha256,
    )

    # Save manifest
    manifest_out = output_dir / "mobile_bundle_manifest.json"
    manifest_out.write_text(json.dumps(asdict(manifest), indent=2))
    logger.info("Manifest saved: %s", manifest_out)

    return manifest


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(manifest: BundleManifest) -> None:
    """Print formatted summary of the mobile bundle."""
    sep = "-" * 78
    print(f"\n{sep}")
    print(f"{'MOBILE BUNDLE SUMMARY':^78}")
    print(sep)
    print(f"  Version:           {manifest.version}")
    print(f"  Total Size:        {manifest.total_size_mb:.1f} MB (uncompressed)")
    print(f"  Compressed:        {manifest.compressed_size_mb:.1f} MB")
    print(f"  Budget:            {manifest.max_size_mb:.0f} MB")
    print(f"  Within Budget:     {'YES' if manifest.within_budget else 'NO -- OVER BUDGET'}")
    print(f"  SHA-256:           {manifest.integrity_sha256[:32]}...")
    print(f"  Built:             {manifest.build_timestamp}")
    print(sep)

    name_w = 20
    size_w = 12
    budget_w = 12
    status_w = 10

    print(
        f"\n  {'Component':<{name_w}} {'Size (MB)':<{size_w}} {'Budget (MB)':<{budget_w}} {'Status':<{status_w}}"
    )
    print(f"  {'-'*name_w} {'-'*size_w} {'-'*budget_w} {'-'*status_w}")

    for comp_dict in manifest.components:
        name = comp_dict["name"]
        size = comp_dict["size_mb"]
        budget = comp_dict["budget_mb"]
        included = comp_dict["included"]

        if not included:
            status = "MISSING"
        elif size > budget:
            status = "OVER"
        else:
            status = "OK"

        print(
            f"  {name:<{name_w}} {size:<{size_w}.1f} {budget:<{budget_w}.1f} {status:<{status_w}}"
        )

    print(sep)

    if not manifest.within_budget:
        overage = manifest.total_size_mb - manifest.max_size_mb
        print(f"\n  BUDGET EXCEEDED by {overage:.1f} MB")
        print("  Recommendations:")
        print("    - Use a more aggressive quantization (Q4_K_S instead of Q4_K_M)")
        print("    - Prune FAISS index (reduce passage count)")
        print("    - Use model distillation for a smaller specialist model")
        print("    - Exclude voice models (--no-voice) for text-only bundle")
    print()


# ---------------------------------------------------------------------------
# Flutter project scaffold generator
# ---------------------------------------------------------------------------


def generate_flutter_scaffold(output_dir: Path) -> None:
    """Generate Flutter project directory structure for offline-first mobile app.

    Creates directory layout and placeholder files for:
    - Offline RAG service integration
    - Bundle management
    - Voice-first mobile UI
    """
    flutter_root = output_dir / "MobileApp"

    dirs = [
        "lib/offline",
        "lib/voice",
        "lib/services",
        "lib/models",
        "lib/screens",
        "lib/widgets",
        "assets/models",
        "assets/voice",
    ]
    for d in dirs:
        (flutter_root / d).mkdir(parents=True, exist_ok=True)

    # Offline RAG service
    (flutter_root / "lib" / "offline" / "offline_rag_service.dart").write_text(
        """import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';

/// Production offline RAG service for on-device vector search.
///
/// Uses FAISS Mobile + ONNX Runtime for fully offline retrieval-augmented
/// generation. Manages bundle lifecycle, delta sync, and integrity verification.
class OfflineRAGService {
  static OfflineRAGService? _instance;
  bool _initialized = false;
  String? _bundlePath;
  String? _bundleVersion;

  OfflineRAGService._();

  static OfflineRAGService get instance {
    _instance ??= OfflineRAGService._();
    return _instance!;
  }

  bool get isInitialized => _initialized;
  String? get bundleVersion => _bundleVersion;

  /// Initialize the offline RAG pipeline with the current bundle.
  Future<void> initialize() async {
    final dir = await getApplicationDocumentsDirectory();
    _bundlePath = '${dir.path}/offline_rag';

    final metaFile = File('$_bundlePath/bundle_manifest.json');
    if (await metaFile.exists()) {
      final meta = jsonDecode(await metaFile.readAsString());
      _bundleVersion = meta['version'] as String?;
      _initialized = true;
    }
  }

  /// Search the local FAISS index for relevant passages.
  Future<List<Map<String, dynamic>>> search(
    String query, {
    int topK = 5,
    double threshold = 0.7,
  }) async {
    if (!_initialized) throw StateError('OfflineRAGService not initialized');

    // TODO: Integrate with ONNX Runtime Mobile for embedding
    // TODO: Integrate with FAISS Mobile for vector search
    // Placeholder: return empty results until native bindings are ready
    return [];
  }

  /// Check if the bundle needs updating.
  Future<bool> needsUpdate(String serverVersion) async {
    if (_bundleVersion == null) return true;
    return _bundleVersion != serverVersion;
  }

  /// Get bundle size in MB.
  Future<double> getBundleSizeMB() async {
    if (_bundlePath == null) return 0.0;
    final dir = Directory(_bundlePath!);
    if (!await dir.exists()) return 0.0;

    int totalBytes = 0;
    await for (final entity in dir.list(recursive: true)) {
      if (entity is File) {
        totalBytes += await entity.length();
      }
    }
    return totalBytes / (1024 * 1024);
  }
}
"""
    )

    # Bundle manager
    (flutter_root / "lib" / "offline" / "bundle_manager.dart").write_text(
        """import 'dart:convert';
import 'dart:io';
import 'package:crypto/crypto.dart';
import 'package:path_provider/path_provider.dart';

/// Manages offline bundle downloads, integrity verification, and delta sync.
///
/// Handles background download of compressed bundles, SHA-256 integrity
/// verification, and incremental delta synchronization for bandwidth efficiency.
class BundleManager {
  static BundleManager? _instance;
  String _serverUrl = '';
  String? _localBundlePath;

  BundleManager._();

  static BundleManager get instance {
    _instance ??= BundleManager._();
    return _instance!;
  }

  /// Configure the bundle manager with server URL.
  void configure({required String serverUrl}) {
    _serverUrl = serverUrl;
  }

  /// Download the latest bundle from the server.
  ///
  /// Shows progress via [onProgress] callback (0.0 to 1.0).
  Future<bool> downloadBundle({
    Function(double)? onProgress,
  }) async {
    final dir = await getApplicationDocumentsDirectory();
    _localBundlePath = '${dir.path}/offline_rag';

    // TODO: Implement streaming download with progress tracking
    // TODO: Implement delta sync (hash-based, only changed chunks)
    // TODO: Verify SHA-256 integrity after download
    return false;
  }

  /// Verify the integrity of the local bundle.
  Future<bool> verifyIntegrity() async {
    if (_localBundlePath == null) return false;

    final manifestFile = File('$_localBundlePath/bundle_manifest.json');
    if (!await manifestFile.exists()) return false;

    // TODO: Verify SHA-256 of each component against manifest
    return true;
  }

  /// Perform delta sync — only download changed chunks.
  Future<DeltaSyncResult> deltaSync() async {
    // TODO: Implement hash-based delta sync
    // 1. Fetch server manifest with chunk hashes
    // 2. Compare with local chunk hashes
    // 3. Download only changed chunks
    // 4. Verify integrity
    return DeltaSyncResult(
      chunksChanged: 0,
      bytesTransferred: 0,
      durationMs: 0,
      success: false,
    );
  }

  /// Get sync status information.
  Future<Map<String, dynamic>> getSyncStatus() async {
    return {
      'localBundlePath': _localBundlePath,
      'serverUrl': _serverUrl,
      'lastSync': null,
      'needsUpdate': true,
    };
  }
}

class DeltaSyncResult {
  final int chunksChanged;
  final int bytesTransferred;
  final int durationMs;
  final bool success;

  DeltaSyncResult({
    required this.chunksChanged,
    required this.bytesTransferred,
    required this.durationMs,
    required this.success,
  });
}
"""
    )

    # Sync service
    (flutter_root / "lib" / "offline" / "sync_service.dart").write_text(
        """import 'dart:async';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'bundle_manager.dart';

/// Background sync service that monitors connectivity and triggers
/// delta sync when network becomes available.
class SyncService {
  static SyncService? _instance;
  StreamSubscription? _connectivitySubscription;
  Timer? _periodicSync;
  bool _isSyncing = false;
  DateTime? _lastSyncTime;

  SyncService._();

  static SyncService get instance {
    _instance ??= SyncService._();
    return _instance!;
  }

  DateTime? get lastSyncTime => _lastSyncTime;
  bool get isSyncing => _isSyncing;

  /// Start monitoring connectivity for auto-sync.
  void startMonitoring({Duration interval = const Duration(hours: 1)}) {
    _connectivitySubscription = Connectivity()
        .onConnectivityChanged
        .listen((result) async {
      if (result.contains(ConnectivityResult.wifi) ||
          result.contains(ConnectivityResult.mobile)) {
        await _trySync();
      }
    });

    _periodicSync = Timer.periodic(interval, (_) => _trySync());
  }

  /// Stop monitoring.
  void stopMonitoring() {
    _connectivitySubscription?.cancel();
    _periodicSync?.cancel();
  }

  Future<void> _trySync() async {
    if (_isSyncing) return;
    _isSyncing = true;

    try {
      final result = await BundleManager.instance.deltaSync();
      if (result.success) {
        _lastSyncTime = DateTime.now();
      }
    } finally {
      _isSyncing = false;
    }
  }

  /// Force an immediate sync attempt.
  Future<bool> forceSync() async {
    await _trySync();
    return _lastSyncTime != null;
  }
}
"""
    )

    # Voice service
    (flutter_root / "lib" / "voice" / "voice_service.dart").write_text(
        """import 'dart:async';
import 'dart:typed_data';

/// Voice-first interface service for offline ASR + TTS.
///
/// Uses Whisper-tiny (ONNX) for speech recognition and Piper/Sherpa
/// for text-to-speech, both running entirely on-device.
class VoiceService {
  static VoiceService? _instance;
  bool _asrReady = false;
  bool _ttsReady = false;

  VoiceService._();

  static VoiceService get instance {
    _instance ??= VoiceService._();
    return _instance!;
  }

  bool get isASRReady => _asrReady;
  bool get isTTSReady => _ttsReady;

  /// Initialize on-device ASR (Whisper-tiny ONNX).
  Future<void> initializeASR({String? modelPath}) async {
    // TODO: Load Whisper-tiny ONNX model via onnxruntime_mobile
    // TODO: Initialize audio preprocessor (mel spectrogram)
    _asrReady = true;
  }

  /// Initialize on-device TTS (Piper/Sherpa ONNX).
  Future<void> initializeTTS({String? modelPath, String language = 'en-ug'}) async {
    // TODO: Load Piper ONNX model for Ugandan English
    // TODO: Initialize audio postprocessor
    _ttsReady = true;
  }

  /// Transcribe audio buffer to text (offline).
  Future<String> transcribe(Float32List audioData, {int sampleRate = 16000}) async {
    if (!_asrReady) throw StateError('ASR not initialized');
    // TODO: Run Whisper-tiny inference
    // TODO: Apply Ugandan English accent adaptation
    return '';
  }

  /// Synthesize text to audio (offline).
  Future<Float32List> synthesize(String text, {double speed = 1.0}) async {
    if (!_ttsReady) throw StateError('TTS not initialized');
    // TODO: Run Piper TTS inference
    // TODO: Apply speech rate adjustment
    return Float32List(0);
  }

  /// Start real-time streaming transcription.
  Stream<String> streamTranscribe() async* {
    // TODO: Implement VAD + chunked Whisper inference
    // TODO: Yield partial transcriptions as they arrive
  }
}
"""
    )

    # Voice chat screen
    (flutter_root / "lib" / "screens" / "voice_chat_screen.dart").write_text(
        """import 'package:flutter/material.dart';

/// Full-screen voice-first chat interface.
///
/// Primary mobile interface designed for low-literacy users in rural areas.
/// Features animated waveform, pulse rings, barge-in support, and
/// offline mode with clear status indicators.
class VoiceChatScreen extends StatefulWidget {
  const VoiceChatScreen({super.key});

  @override
  State<VoiceChatScreen> createState() => _VoiceChatScreenState();
}

class _VoiceChatScreenState extends State<VoiceChatScreen>
    with TickerProviderStateMixin {
  // Voice states: idle, listening, processing, speaking
  String _voiceState = 'idle';
  bool _isOffline = false;
  String _transcript = '';
  String _response = '';
  late AnimationController _pulseController;
  late AnimationController _waveController;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat();
    _waveController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
    );
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _waveController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            // Offline banner
            if (_isOffline)
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                color: Colors.amber.shade800,
                child: const Row(
                  children: [
                    Icon(Icons.cloud_off, color: Colors.white, size: 16),
                    SizedBox(width: 8),
                    Text(
                      'Offline Mode - Sync when online',
                      style: TextStyle(color: Colors.white, fontSize: 12),
                    ),
                  ],
                ),
              ),

            // Status indicator
            Padding(
              padding: const EdgeInsets.all(16),
              child: Text(
                _voiceState == 'listening' ? 'Listening...'
                    : _voiceState == 'processing' ? 'Thinking...'
                    : _voiceState == 'speaking' ? 'Speaking...'
                    : 'Tap to speak',
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 16,
                  fontWeight: FontWeight.w300,
                ),
              ),
            ),

            // Conversation area
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    if (_transcript.isNotEmpty)
                      Text(
                        _transcript,
                        style: const TextStyle(color: Colors.white, fontSize: 18),
                        textAlign: TextAlign.center,
                      ),
                    const SizedBox(height: 24),
                    if (_response.isNotEmpty)
                      Text(
                        _response,
                        style: TextStyle(
                          color: Colors.tealAccent.shade200,
                          fontSize: 16,
                        ),
                        textAlign: TextAlign.center,
                      ),
                  ],
                ),
              ),
            ),

            // Waveform + mic button area
            SizedBox(
              height: 200,
              child: Center(
                child: GestureDetector(
                  onTap: _toggleListening,
                  child: AnimatedBuilder(
                    animation: _pulseController,
                    builder: (context, child) {
                      return Container(
                        width: 80 + (_voiceState == 'listening' ? 20 * _pulseController.value : 0),
                        height: 80 + (_voiceState == 'listening' ? 20 * _pulseController.value : 0),
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: _voiceState == 'listening'
                              ? Colors.tealAccent
                              : Colors.white24,
                        ),
                        child: Icon(
                          _voiceState == 'listening' ? Icons.mic : Icons.mic_none,
                          color: Colors.white,
                          size: 36,
                        ),
                      );
                    },
                  ),
                ),
              ),
            ),

            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }

  void _toggleListening() {
    setState(() {
      _voiceState = _voiceState == 'listening' ? 'idle' : 'listening';
    });
  }
}
"""
    )

    # Waveform widget
    (flutter_root / "lib" / "widgets" / "waveform_widget.dart").write_text(
        """import 'dart:math';
import 'package:flutter/material.dart';

/// Animated waveform visualization for voice-first interface.
///
/// Renders a real-time audio waveform using CustomPainter with
/// smooth animations and mode-responsive styling.
class WaveformWidget extends StatelessWidget {
  final List<double> amplitudes;
  final Color color;
  final double height;
  final bool isActive;

  const WaveformWidget({
    super.key,
    required this.amplitudes,
    this.color = Colors.tealAccent,
    this.height = 100,
    this.isActive = false,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: height,
      width: double.infinity,
      child: CustomPaint(
        painter: _WaveformPainter(
          amplitudes: amplitudes,
          color: color,
          isActive: isActive,
        ),
      ),
    );
  }
}

class _WaveformPainter extends CustomPainter {
  final List<double> amplitudes;
  final Color color;
  final bool isActive;

  _WaveformPainter({
    required this.amplitudes,
    required this.color,
    required this.isActive,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color.withValues(alpha: isActive ? 0.8 : 0.3)
      ..strokeWidth = 2.0
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final centerY = size.height / 2;
    final barWidth = size.width / max(amplitudes.length, 1);

    for (int i = 0; i < amplitudes.length; i++) {
      final amp = amplitudes[i].clamp(0.0, 1.0);
      final barHeight = amp * size.height * 0.8;
      final x = i * barWidth + barWidth / 2;

      canvas.drawLine(
        Offset(x, centerY - barHeight / 2),
        Offset(x, centerY + barHeight / 2),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _WaveformPainter oldDelegate) {
    return oldDelegate.amplitudes != amplitudes ||
        oldDelegate.isActive != isActive;
  }
}
"""
    )

    logger.info("Flutter scaffold generated at %s", flutter_root)
    logger.info("  Directories: %s", ", ".join(dirs))
    logger.info(
        "  Files: 6 Dart files (offline RAG, bundle manager, sync, voice, screens, widgets)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export and validate mobile bundle for Flutter offline-first deployment.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mobile_bundle",
        help="Output directory for the bundle (default: outputs/mobile_bundle)",
    )
    parser.add_argument(
        "--max-bundle-mb",
        type=float,
        default=MAX_BUNDLE_SIZE_MB,
        help=f"Maximum total bundle size in MB (default: {MAX_BUNDLE_SIZE_MB})",
    )
    parser.add_argument(
        "--include-voice",
        action="store_true",
        help="Include voice models (Whisper-tiny + Piper TTS) in bundle",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Semantic version for the bundle (default: 1.0.0)",
    )
    parser.add_argument(
        "--generate-flutter",
        action="store_true",
        help="Generate Flutter project scaffold in output directory",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip archive creation (just discover + validate)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir

    # 1. Discover components
    print("\n[1/4] Discovering bundle components ...")
    components = discover_components(include_voice=args.include_voice)

    included = [c for c in components if c.included]
    missing = [c for c in components if not c.included]
    logger.info(
        "Found %d components (%d included, %d missing)",
        len(components),
        len(included),
        len(missing),
    )

    for c in missing:
        logger.warning("Missing component: %s (budget: %.0f MB)", c.name, c.budget_mb)

    # 2. Validate sizes
    print("[2/4] Validating component sizes ...")
    total_size = sum(c.size_mb for c in included)
    logger.info(
        "Total uncompressed size: %.1f MB (budget: %.0f MB)", total_size, args.max_bundle_mb
    )

    # 3. Build bundle
    if not args.no_archive and included:
        print(f"[3/4] Building bundle v{args.version} ...")
        manifest = build_bundle(
            components=components,
            output_dir=output_dir,
            max_size_mb=args.max_bundle_mb,
            version=args.version,
        )
        print_summary(manifest)

        if not manifest.within_budget:
            logger.error(
                "BUNDLE SIZE EXCEEDED: %.1f MB > %.0f MB limit",
                manifest.total_size_mb,
                manifest.max_size_mb,
            )
            sys.exit(1)
    else:
        print("[3/4] Skipping archive creation.")
        manifest = BundleManifest(
            version=args.version,
            total_size_mb=round(total_size, 2),
            max_size_mb=args.max_bundle_mb,
            within_budget=total_size <= args.max_bundle_mb,
            components=[asdict(c) for c in components],
        )
        print_summary(manifest)

    # 4. Generate Flutter scaffold
    if args.generate_flutter:
        print("[4/4] Generating Flutter project scaffold ...")
        generate_flutter_scaffold(output_dir)
    else:
        print("[4/4] Skipping Flutter scaffold (use --generate-flutter to create).")

    print(f"\nMobile bundle export complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
