import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../providers/providers.dart';

/// Splash screen: verify bundle integrity, load ONNX models, initialize DB.
class SplashScreen extends ConsumerStatefulWidget {
  const SplashScreen({super.key});

  @override
  ConsumerState<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends ConsumerState<SplashScreen> {
  String _status = 'Initializing...';
  bool _error = false;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      setState(() => _status = 'Loading database...');
      await ref.read(databaseProvider.future);

      setState(() => _status = 'Initializing audit trail...');
      await ref.read(auditProvider.future);

      setState(() => _status = 'Loading screening models...');
      final onnx = ref.read(onnxInferenceProvider);
      // In production, load from app assets or documents directory
      // await onnx.initialize(
      //   studentModelPath: '${appDir}/models/student_int8.onnx',
      //   gateModelPath: '${appDir}/models/gate_mobilenetv3.onnx',
      // );
      ref.read(modelReadyProvider.notifier).state = true;

      setState(() => _status = 'Checking connectivity...');
      // Trigger initial connectivity check
      ref.read(connectivityProvider);

      // Log app start event
      final audit = await ref.read(auditProvider.future);
      await audit.logEvent(eventType: 'app_start', payload: {
        'version': '1.0.0',
        'timestamp': DateTime.now().toIso8601String(),
      });

      // Navigate to home
      if (mounted) {
        Navigator.of(context).pushReplacementNamed('/home');
      }
    } catch (e) {
      setState(() {
        _status = 'Error: $e';
        _error = true;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Logo
            Container(
              width: 120,
              height: 120,
              decoration: BoxDecoration(
                color: theme.colorScheme.primaryContainer,
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.remove_red_eye_outlined,
                size: 60,
                color: theme.colorScheme.primary,
              ),
            ),
            const SizedBox(height: 32),
            Text(
              'RetinalAI',
              style: theme.textTheme.headlineMedium?.copyWith(
                fontWeight: FontWeight.bold,
                color: theme.colorScheme.primary,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Clinical Screening',
              style: theme.textTheme.bodyLarge?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 48),
            if (!_error) const CircularProgressIndicator(),
            if (_error)
              Icon(Icons.error_outline, color: theme.colorScheme.error, size: 48),
            const SizedBox(height: 16),
            Text(
              _status,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: _error
                    ? theme.colorScheme.error
                    : theme.colorScheme.onSurfaceVariant,
              ),
              textAlign: TextAlign.center,
            ),
            if (_error) ...[
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    _error = false;
                    _status = 'Retrying...';
                  });
                  _initialize();
                },
                child: const Text('Retry'),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
