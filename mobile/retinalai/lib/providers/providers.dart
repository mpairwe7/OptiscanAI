import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../data/database/app_database.dart';
import '../services/audit_service.dart';
import '../services/connectivity_service.dart';
import '../services/fundus_gate_service.dart';
import '../services/onnx_inference_service.dart';
import '../services/sync_service.dart';

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------
final databaseProvider = FutureProvider<AppDatabase>((ref) async {
  return AppDatabase.getInstance();
});

// ---------------------------------------------------------------------------
// Core services
// ---------------------------------------------------------------------------
final connectivityProvider = Provider<ConnectivityService>((ref) {
  final svc = ConnectivityService(serverBaseUrl: 'http://localhost:8080');
  ref.onDispose(() => svc.dispose());
  return svc;
});

final connectivityStateProvider = StreamProvider<ConnectivityState>((ref) {
  return ref.watch(connectivityProvider).stateStream;
});

final isOnlineProvider = Provider<bool>((ref) {
  final state = ref.watch(connectivityStateProvider);
  return state.valueOrNull == ConnectivityState.online;
});

final onnxInferenceProvider = Provider<OnnxInferenceService>((ref) {
  return OnnxInferenceService();
});

final fundusGateProvider = Provider<FundusGateService>((ref) {
  return FundusGateService();
});

final auditProvider = FutureProvider<AuditService>((ref) async {
  final db = await ref.watch(databaseProvider.future);
  final svc = AuditService(db);
  await svc.initialize();
  return svc;
});

final syncProvider = FutureProvider<SyncService>((ref) async {
  final db = await ref.watch(databaseProvider.future);
  final connectivity = ref.watch(connectivityProvider);
  return SyncService(
    db: db,
    connectivity: connectivity,
    serverBaseUrl: 'http://localhost:8080',
  );
});

// ---------------------------------------------------------------------------
// Inference state
// ---------------------------------------------------------------------------

/// Tracks whether ONNX models are loaded and ready.
final modelReadyProvider = StateProvider<bool>((ref) => false);

/// Holds the most recent inference result for display.
final lastInferenceResultProvider = StateProvider<InferenceResult?>((ref) => null);

// ---------------------------------------------------------------------------
// Sync status
// ---------------------------------------------------------------------------
final syncStatusProvider = FutureProvider<SyncStatusSummary>((ref) async {
  final sync = await ref.watch(syncProvider.future);
  return sync.getStatus();
});
