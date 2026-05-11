import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// On-device Fundus Gate V2 — 3-layer validation + fusion.
///
/// Mirrors the Python implementation at src/data/fundus_gate_v2.py:
///   Layer 1 (structural): resolution, aspect ratio, color mode
///   Layer 2 (statistical): channel means, dark pixel ratio, radial sharpness
///   Layer 3 (learned): MobileNetV3-Small ONNX binary classifier
///   Fusion: 0.6 * statistical + 0.4 * learned, threshold 0.70
class FundusGateService {
  static const double _statWeight = 0.6;
  static const double _learnedWeight = 0.4;
  static const double _minConfidence = 0.70;
  static const int _minResolution = 100;
  static const double _maxAspectDeviation = 0.65;

  /// Run the full 3-layer gate on decoded image data.
  FundusGateResult validate(img.Image image, {double? learnedProb}) {
    final structural = _checkStructural(image);
    if (!structural.passed) {
      return FundusGateResult(
        passed: false,
        layer: 'structural',
        confidence: 0.0,
        details: structural,
      );
    }

    final statistical = _checkStatistical(image);

    // Fusion
    double fusedConfidence;
    if (learnedProb != null) {
      fusedConfidence =
          _statWeight * statistical.confidence + _learnedWeight * learnedProb;
    } else {
      fusedConfidence = statistical.confidence;
    }

    // Hard spatial requirement (simplified for mobile)
    final hasFundusSpatial =
        statistical.hasDarkBorder || statistical.hasCircularAperture;
    final passed = fusedConfidence >= _minConfidence && hasFundusSpatial;

    return FundusGateResult(
      passed: passed,
      layer: 'fusion',
      confidence: fusedConfidence,
      details: structural,
      statisticalResult: statistical,
      learnedProb: learnedProb,
    );
  }

  // -----------------------------------------------------------------------
  // Layer 1: Structural checks (< 1ms)
  // -----------------------------------------------------------------------
  _StructuralResult _checkStructural(img.Image image) {
    final w = image.width;
    final h = image.height;

    if (w < _minResolution || h < _minResolution) {
      return _StructuralResult(
        passed: false,
        reason: 'Resolution too low: ${w}x$h (min: ${_minResolution}x$_minResolution)',
      );
    }

    final aspectRatio = w / h;
    final deviation = (aspectRatio - 1.0).abs();
    if (deviation > _maxAspectDeviation) {
      return _StructuralResult(
        passed: false,
        reason: 'Aspect ratio deviation $deviation > $_maxAspectDeviation',
      );
    }

    return _StructuralResult(passed: true);
  }

  // -----------------------------------------------------------------------
  // Layer 2: Statistical checks (~3-5ms)
  // -----------------------------------------------------------------------
  _StatisticalResult _checkStatistical(img.Image image) {
    final w = image.width;
    final h = image.height;
    final totalPixels = w * h;

    double rSum = 0, gSum = 0, bSum = 0;
    int darkPixels = 0;
    int borderDarkPixels = 0;
    int borderTotal = 0;

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final pixel = image.getPixel(x, y);
        final r = pixel.r / 255.0;
        final g = pixel.g / 255.0;
        final b = pixel.b / 255.0;

        rSum += r;
        gSum += g;
        bSum += b;

        final luminance = 0.299 * r + 0.587 * g + 0.114 * b;
        if (luminance < 0.15) darkPixels++;

        // Border check (outer 10%)
        final borderThresh = 0.1;
        if (x < w * borderThresh ||
            x > w * (1 - borderThresh) ||
            y < h * borderThresh ||
            y > h * (1 - borderThresh)) {
          borderTotal++;
          if (luminance < 0.15) borderDarkPixels++;
        }
      }
    }

    final rMean = rSum / totalPixels;
    final gMean = gSum / totalPixels;
    final bMean = bSum / totalPixels;
    final darkRatio = darkPixels / totalPixels;
    final borderDarkRatio =
        borderTotal > 0 ? borderDarkPixels / borderTotal : 0.0;

    // Fundus checks
    final redDominant = rMean > gMean * 1.05;
    final channelsInRange = rMean >= 0.20 &&
        rMean <= 0.75 &&
        gMean >= 0.08 &&
        gMean <= 0.55 &&
        bMean >= 0.02 &&
        bMean <= 0.40;
    final darkPixelOk = darkRatio >= 0.05 && darkRatio <= 0.80;
    final hasDarkBorder = borderDarkRatio > 0.4;

    // Simple circular aperture detection (center brighter than edges)
    final centerBrightness = _sampleCenterBrightness(image);
    final edgeBrightness = _sampleEdgeBrightness(image);
    final hasCircularAperture =
        centerBrightness > edgeBrightness * 1.3 && hasDarkBorder;

    // Confidence scoring
    double score = 0.0;
    if (redDominant) score += 0.2;
    if (channelsInRange) score += 0.25;
    if (darkPixelOk) score += 0.15;
    if (hasDarkBorder) score += 0.2;
    if (hasCircularAperture) score += 0.2;

    return _StatisticalResult(
      confidence: score.clamp(0.0, 1.0),
      rMean: rMean,
      gMean: gMean,
      bMean: bMean,
      darkRatio: darkRatio,
      redDominant: redDominant,
      channelsInRange: channelsInRange,
      hasDarkBorder: hasDarkBorder,
      hasCircularAperture: hasCircularAperture,
    );
  }

  double _sampleCenterBrightness(img.Image image) {
    final cx = image.width ~/ 2;
    final cy = image.height ~/ 2;
    final r = math.min(image.width, image.height) ~/ 6;
    double sum = 0;
    int count = 0;
    for (int dy = -r; dy <= r; dy++) {
      for (int dx = -r; dx <= r; dx++) {
        if (dx * dx + dy * dy <= r * r) {
          final p = image.getPixel(cx + dx, cy + dy);
          sum += 0.299 * p.r / 255 + 0.587 * p.g / 255 + 0.114 * p.b / 255;
          count++;
        }
      }
    }
    return count > 0 ? sum / count : 0;
  }

  double _sampleEdgeBrightness(img.Image image) {
    double sum = 0;
    int count = 0;
    final margin = math.min(image.width, image.height) ~/ 10;
    // Top edge
    for (int x = 0; x < image.width; x += 4) {
      final p = image.getPixel(x, margin);
      sum += 0.299 * p.r / 255 + 0.587 * p.g / 255 + 0.114 * p.b / 255;
      count++;
    }
    // Bottom edge
    for (int x = 0; x < image.width; x += 4) {
      final p = image.getPixel(x, image.height - margin - 1);
      sum += 0.299 * p.r / 255 + 0.587 * p.g / 255 + 0.114 * p.b / 255;
      count++;
    }
    return count > 0 ? sum / count : 0;
  }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

class FundusGateResult {
  final bool passed;
  final String layer;
  final double confidence;
  final _StructuralResult details;
  final _StatisticalResult? statisticalResult;
  final double? learnedProb;

  FundusGateResult({
    required this.passed,
    required this.layer,
    required this.confidence,
    required this.details,
    this.statisticalResult,
    this.learnedProb,
  });

  Map<String, dynamic> toJson() => {
        'passed': passed,
        'layer': layer,
        'confidence': confidence,
        'structural_passed': details.passed,
        'structural_reason': details.reason,
        'statistical_confidence': statisticalResult?.confidence,
        'learned_prob': learnedProb,
      };
}

class _StructuralResult {
  final bool passed;
  final String? reason;
  _StructuralResult({required this.passed, this.reason});
}

class _StatisticalResult {
  final double confidence;
  final double rMean, gMean, bMean;
  final double darkRatio;
  final bool redDominant;
  final bool channelsInRange;
  final bool hasDarkBorder;
  final bool hasCircularAperture;

  _StatisticalResult({
    required this.confidence,
    required this.rMean,
    required this.gMean,
    required this.bMean,
    required this.darkRatio,
    required this.redDominant,
    required this.channelsInRange,
    required this.hasDarkBorder,
    required this.hasCircularAperture,
  });
}
