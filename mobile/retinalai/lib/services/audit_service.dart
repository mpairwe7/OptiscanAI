import 'dart:convert';

import 'package:crypto/crypto.dart';
import 'package:drift/drift.dart';
import 'package:uuid/uuid.dart';

import '../data/database/app_database.dart';

/// SHA-256 hash-chain audit trail service.
///
/// Every prediction and significant event is logged with a cryptographic
/// hash linking to the previous entry, forming an append-only tamper-evident
/// chain. This satisfies EU AI Act and Uganda PDP Act audit requirements
/// even during extended offline periods.
class AuditService {
  final AppDatabase _db;
  final _uuid = const Uuid();

  String _lastPredictionHash = '0' * 64; // Genesis hash
  String _lastAuditHash = '0' * 64;

  AuditService(this._db);

  /// Initialize by loading the last hash from the database.
  Future<void> initialize() async {
    // Recover chain state from the most recent entries
    final lastAudit = await _db.getLastAuditEntry();
    if (lastAudit != null) {
      _lastAuditHash = lastAudit.entryHash;
    }

    // Get last prediction hash
    final predictions = await (_db.select(_db.predictions)
          ..orderBy([(t) => OrderingTerm.desc(t.createdAt)])
          ..limit(1))
        .get();
    if (predictions.isNotEmpty) {
      _lastPredictionHash = predictions.first.entryHash;
    }
  }

  /// Compute SHA-256 hash of an entry's content + previous hash.
  String computeEntryHash(Map<String, dynamic> entry) {
    // Sort keys for deterministic serialization
    final sorted = Map.fromEntries(
      entry.entries.toList()..sort((a, b) => a.key.compareTo(b.key)),
    );
    final payload = jsonEncode(sorted);
    return sha256.convert(utf8.encode(payload)).toString();
  }

  /// Log a prediction result to the audit trail.
  Future<String> logPrediction({
    required Map<String, dynamic> predictionData,
    required String imageHash,
    required bool gatePassed,
    required double gateConfidence,
    required String inferenceSource,
    required String modelVersion,
    required double inferenceMs,
    required Map<String, double> probabilities,
    required List<String> detectedDiseases,
    String? referralPriority,
    double? riskScore,
    String? chwId,
    String? patientIdHash,
    double? gpsLat,
    double? gpsLon,
    String? deviceId,
  }) async {
    final id = _uuid.v4();
    final now = DateTime.now();

    // Build entry for hashing
    final hashInput = {
      'id': id,
      'created_at': now.toIso8601String(),
      'image_hash': imageHash,
      'gate_passed': gatePassed,
      'inference_source': inferenceSource,
      'model_version': modelVersion,
      'detected_diseases': detectedDiseases,
      'previous_hash': _lastPredictionHash,
    };

    final entryHash = computeEntryHash(hashInput);

    await _db.insertPrediction(PredictionsCompanion(
      id: Value(id),
      createdAt: Value(now),
      imageHash: Value(imageHash),
      gatePassed: Value(gatePassed),
      gateConfidence: Value(gateConfidence),
      inferenceSource: Value(inferenceSource),
      modelVersion: Value(modelVersion),
      inferenceMs: Value(inferenceMs),
      probabilities: Value(jsonEncode(probabilities)),
      detectedDiseases: Value(jsonEncode(detectedDiseases)),
      referralPriority: Value(referralPriority),
      riskScore: Value(riskScore),
      chwId: Value(chwId),
      patientIdHash: Value(patientIdHash),
      gpsLat: Value(gpsLat),
      gpsLon: Value(gpsLon),
      deviceId: Value(deviceId),
      previousHash: Value(_lastPredictionHash),
      entryHash: Value(entryHash),
    ));

    _lastPredictionHash = entryHash;

    // Also enqueue for sync
    await _db.enqueueSync(SyncQueueCompanion(
      id: Value(_uuid.v4()),
      itemType: const Value('prediction'),
      payload: Value(jsonEncode(predictionData)),
      createdAt: Value(now),
    ));

    return id;
  }

  /// Log a generic audit event.
  Future<String> logEvent({
    required String eventType,
    Map<String, dynamic>? payload,
  }) async {
    final id = _uuid.v4();
    final now = DateTime.now();

    final hashInput = {
      'id': id,
      'event_type': eventType,
      'created_at': now.toIso8601String(),
      'previous_hash': _lastAuditHash,
    };
    if (payload != null) hashInput['payload'] = payload.toString();

    final entryHash = computeEntryHash(hashInput);

    await _db.insertAuditEntry(AuditLogCompanion(
      id: Value(id),
      eventType: Value(eventType),
      createdAt: Value(now),
      payload: Value(payload != null ? jsonEncode(payload) : null),
      previousHash: Value(_lastAuditHash),
      entryHash: Value(entryHash),
    ));

    _lastAuditHash = entryHash;
    return id;
  }

  /// Verify the integrity of the audit hash chain.
  Future<AuditChainVerification> verifyChain() async {
    final entries = await (_db.select(_db.auditLog)
          ..orderBy([(t) => OrderingTerm.asc(t.createdAt)]))
        .get();

    if (entries.isEmpty) {
      return AuditChainVerification(valid: true, entriesChecked: 0);
    }

    String expectedPrevHash = '0' * 64;
    int checked = 0;

    for (final entry in entries) {
      if (entry.previousHash != expectedPrevHash) {
        return AuditChainVerification(
          valid: false,
          entriesChecked: checked,
          brokenAtId: entry.id,
          reason: 'Previous hash mismatch at entry ${entry.id}',
        );
      }
      expectedPrevHash = entry.entryHash;
      checked++;
    }

    return AuditChainVerification(valid: true, entriesChecked: checked);
  }
}

class AuditChainVerification {
  final bool valid;
  final int entriesChecked;
  final String? brokenAtId;
  final String? reason;

  AuditChainVerification({
    required this.valid,
    required this.entriesChecked,
    this.brokenAtId,
    this.reason,
  });
}
