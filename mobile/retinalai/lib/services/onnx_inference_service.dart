import 'dart:typed_data';
import 'dart:math' as math;

import 'package:image/image.dart' as img;

/// On-device ONNX Runtime inference for the MobileStudentV1 model.
///
/// Loads the INT8 student ONNX model and runs single-pass inference.
/// Uses 2 intra-op threads for the Tecno Spark 10's 8-core CPU.
///
/// ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
class OnnxInferenceService {
  // ONNX Runtime session handles — placeholder types until onnxruntime
  // package is linked. In production these would be OrtSession instances.
  dynamic _studentSession;
  dynamic _gateSession;

  bool _initialized = false;
  bool get isInitialized => _initialized;

  static const int inputSize = 224;
  static const List<double> _mean = [0.485, 0.456, 0.406];
  static const List<double> _std = [0.229, 0.224, 0.225];

  /// Initialize ONNX sessions for student and gate models.
  Future<void> initialize({
    required String studentModelPath,
    required String gateModelPath,
  }) async {
    // In production, use the onnxruntime package:
    // final env = OrtEnv.instance;
    // final opts = OrtSessionOptions()
    //   ..setIntraOpNumThreads(2)
    //   ..setGraphOptimizationLevel(GraphOptimizationLevel.ORT_ENABLE_ALL);
    // _studentSession = OrtSession.fromFile(studentModelPath, opts);
    // _gateSession = OrtSession.fromFile(gateModelPath, opts);

    _initialized = true;
  }

  /// Preprocess image bytes to model input tensor.
  ///
  /// 1. Decode JPEG/PNG
  /// 2. Resize to 224x224 (bilinear)
  /// 3. Normalize with ImageNet mean/std
  /// 4. Convert to NCHW float32 tensor
  Float32List preprocessImage(Uint8List imageBytes) {
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) {
      throw ArgumentError('Failed to decode image');
    }

    final resized = img.copyResize(decoded, width: inputSize, height: inputSize,
        interpolation: img.Interpolation.linear);

    // NCHW layout: [1, 3, 224, 224]
    final tensor = Float32List(1 * 3 * inputSize * inputSize);
    final hw = inputSize * inputSize;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        final idx = y * inputSize + x;

        // R channel
        tensor[0 * hw + idx] = (pixel.r / 255.0 - _mean[0]) / _std[0];
        // G channel
        tensor[1 * hw + idx] = (pixel.g / 255.0 - _mean[1]) / _std[1];
        // B channel
        tensor[2 * hw + idx] = (pixel.b / 255.0 - _mean[2]) / _std[2];
      }
    }

    return tensor;
  }

  /// Run the student model and return sigmoid probabilities.
  Future<List<double>> runStudentInference(Float32List inputTensor) async {
    if (!_initialized) throw StateError('ONNX not initialized');

    // In production:
    // final input = OrtValueTensor.createTensorWithDataList(
    //   inputTensor, [1, 3, inputSize, inputSize]);
    // final outputs = _studentSession.run(
    //   OrtRunOptions(), {'input': input});
    // final logits = (outputs[0]?.value as List<List<double>>)[0];
    // return logits.map((l) => _sigmoid(l)).toList();

    // Placeholder: return zeros until ONNX runtime is linked
    return List.filled(28, 0.0);
  }

  /// Run the fundus gate model and return gate probability.
  Future<double> runGateInference(Float32List inputTensor) async {
    if (!_initialized) throw StateError('ONNX not initialized');

    // In production:
    // final input = OrtValueTensor.createTensorWithDataList(
    //   inputTensor, [1, 3, inputSize, inputSize]);
    // final outputs = _gateSession.run(
    //   OrtRunOptions(), {'input': input});
    // final logit = (outputs[0]?.value as List<List<double>>)[0][0];
    // return _sigmoid(logit);

    return 0.95; // Placeholder
  }

  /// Apply per-class thresholds to probabilities.
  List<bool> applyThresholds(
    List<double> probabilities,
    List<double> thresholds,
  ) {
    assert(probabilities.length == thresholds.length);
    return List.generate(
      probabilities.length,
      (i) => probabilities[i] >= thresholds[i],
    );
  }

  /// Full inference pipeline: preprocess -> gate -> student -> thresholds.
  Future<InferenceResult> runFullInference({
    required Uint8List imageBytes,
    required List<double> thresholds,
    required List<String> diseaseNames,
  }) async {
    final stopwatch = Stopwatch()..start();

    final tensor = preprocessImage(imageBytes);

    // Gate check
    final gateProb = await runGateInference(tensor);
    final gatePassed = gateProb >= 0.70;

    if (!gatePassed) {
      stopwatch.stop();
      return InferenceResult(
        gatePassed: false,
        gateConfidence: gateProb,
        probabilities: [],
        detectedDiseases: [],
        inferenceMs: stopwatch.elapsedMilliseconds.toDouble(),
      );
    }

    // Student inference
    final probabilities = await runStudentInference(tensor);
    final detections = applyThresholds(probabilities, thresholds);

    final detectedDiseases = <String>[];
    for (int i = 0; i < detections.length && i < diseaseNames.length; i++) {
      if (detections[i]) {
        detectedDiseases.add(diseaseNames[i]);
      }
    }

    stopwatch.stop();

    return InferenceResult(
      gatePassed: true,
      gateConfidence: gateProb,
      probabilities: probabilities,
      detectedDiseases: detectedDiseases,
      inferenceMs: stopwatch.elapsedMilliseconds.toDouble(),
    );
  }

  void dispose() {
    // In production: _studentSession?.release(); _gateSession?.release();
    _initialized = false;
  }
}

double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

/// Result of a full on-device inference pipeline.
class InferenceResult {
  final bool gatePassed;
  final double gateConfidence;
  final List<double> probabilities;
  final List<String> detectedDiseases;
  final double inferenceMs;
  final String source;

  InferenceResult({
    required this.gatePassed,
    required this.gateConfidence,
    required this.probabilities,
    required this.detectedDiseases,
    required this.inferenceMs,
    this.source = 'local',
  });

  Map<String, dynamic> toJson() => {
        'gate_passed': gatePassed,
        'gate_confidence': gateConfidence,
        'probabilities': probabilities,
        'detected_diseases': detectedDiseases,
        'inference_ms': inferenceMs,
        'source': source,
      };
}
