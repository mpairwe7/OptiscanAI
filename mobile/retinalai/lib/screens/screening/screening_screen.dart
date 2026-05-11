import 'dart:convert';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../providers/providers.dart';
import '../../services/onnx_inference_service.dart';
import '../../widgets/offline_banner.dart';

/// Screening screen: runs gate + inference + displays results.
class ScreeningScreen extends ConsumerStatefulWidget {
  const ScreeningScreen({super.key});

  @override
  ConsumerState<ScreeningScreen> createState() => _ScreeningScreenState();
}

class _ScreeningScreenState extends ConsumerState<ScreeningScreen> {
  InferenceResult? _result;
  bool _isProcessing = true;
  String _status = 'Running fundus gate...';

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final imageBytes = ModalRoute.of(context)?.settings.arguments as Uint8List?;
    if (imageBytes != null && _result == null) {
      _runScreening(imageBytes);
    }
  }

  Future<void> _runScreening(Uint8List imageBytes) async {
    try {
      setState(() {
        _isProcessing = true;
        _status = 'Running fundus gate...';
      });

      final onnx = ref.read(onnxInferenceProvider);

      // Default thresholds and disease names (loaded from bundle in production)
      final thresholds = List.filled(28, 0.5);
      final diseaseNames = [
        'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ERM', 'LS',
        'MS', 'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST',
        'AION', 'PT', 'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'CWS',
      ];

      setState(() => _status = 'Running inference...');

      final result = await onnx.runFullInference(
        imageBytes: imageBytes,
        thresholds: thresholds,
        diseaseNames: diseaseNames,
      );

      setState(() {
        _result = result;
        _isProcessing = false;
      });

      // Log to audit trail
      final audit = await ref.read(auditProvider.future);
      final imageHash = sha256.convert(imageBytes).toString();

      final probMap = <String, double>{};
      for (int i = 0; i < result.probabilities.length && i < diseaseNames.length; i++) {
        probMap[diseaseNames[i]] = result.probabilities[i];
      }

      await audit.logPrediction(
        predictionData: result.toJson(),
        imageHash: imageHash,
        gatePassed: result.gatePassed,
        gateConfidence: result.gateConfidence,
        inferenceSource: result.source,
        modelVersion: '1.0.0',
        inferenceMs: result.inferenceMs,
        probabilities: probMap,
        detectedDiseases: result.detectedDiseases,
      );

      ref.read(lastInferenceResultProvider.notifier).state = result;
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _status = 'Error: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isOnline = ref.watch(isOnlineProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Screening Results')),
      body: Column(
        children: [
          if (!isOnline) const OfflineBanner(),
          Expanded(
            child: _isProcessing
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const CircularProgressIndicator(),
                        const SizedBox(height: 16),
                        Text(_status),
                      ],
                    ),
                  )
                : _result == null
                    ? const Center(child: Text('No image provided'))
                    : _buildResults(theme),
          ),
        ],
      ),
    );
  }

  Widget _buildResults(ThemeData theme) {
    final result = _result!;

    if (!result.gatePassed) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.warning_amber_rounded,
                  size: 72, color: theme.colorScheme.error),
              const SizedBox(height: 16),
              Text('Image Rejected',
                  style: theme.textTheme.headlineSmall
                      ?.copyWith(color: theme.colorScheme.error)),
              const SizedBox(height: 8),
              Text(
                'The image does not appear to be a fundus photograph.\n'
                'Gate confidence: ${(result.gateConfidence * 100).toStringAsFixed(1)}%',
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              FilledButton.icon(
                onPressed: () => Navigator.pushReplacementNamed(context, '/capture'),
                icon: const Icon(Icons.camera_alt),
                label: const Text('Retake Image'),
              ),
            ],
          ),
        ),
      );
    }

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Summary card
          Card(
            color: result.detectedDiseases.isEmpty
                ? Colors.green.shade50
                : Colors.orange.shade50,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  Icon(
                    result.detectedDiseases.isEmpty
                        ? Icons.check_circle
                        : Icons.warning_amber_rounded,
                    size: 48,
                    color: result.detectedDiseases.isEmpty
                        ? Colors.green
                        : Colors.orange,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    result.detectedDiseases.isEmpty
                        ? 'No Diseases Detected'
                        : '${result.detectedDiseases.length} Finding(s)',
                    style: theme.textTheme.titleLarge,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${result.source.toUpperCase()} | '
                    '${result.inferenceMs.toStringAsFixed(0)} ms | '
                    'Gate: ${(result.gateConfidence * 100).toStringAsFixed(0)}%',
                    style: theme.textTheme.bodySmall,
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Disease findings
          if (result.detectedDiseases.isNotEmpty) ...[
            Text('Detected Conditions', style: theme.textTheme.titleMedium),
            const SizedBox(height: 8),
            ...result.detectedDiseases.map((d) => Card(
                  child: ListTile(
                    leading: const Icon(Icons.medical_information,
                        color: Colors.orange),
                    title: Text(d),
                    trailing: Text(
                      result.probabilities.isNotEmpty
                          ? '${(result.probabilities[result.detectedDiseases.indexOf(d)] * 100).toStringAsFixed(1)}%'
                          : '',
                    ),
                  ),
                )),
            const SizedBox(height: 16),
          ],

          // Actions
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () =>
                      Navigator.pushReplacementNamed(context, '/capture'),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('New Scan'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: FilledButton.icon(
                  onPressed: () => Navigator.pushReplacementNamed(context, '/home'),
                  icon: const Icon(Icons.home),
                  label: const Text('Home'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
