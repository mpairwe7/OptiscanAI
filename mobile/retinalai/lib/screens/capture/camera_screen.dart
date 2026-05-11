import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Camera screen for fundus image capture.
///
/// Uses Camera2 API (no Google Play Services dependency).
/// Shows a circle overlay to guide fundus positioning.
class CameraScreen extends ConsumerStatefulWidget {
  const CameraScreen({super.key});

  @override
  ConsumerState<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends ConsumerState<CameraScreen> {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isInitialized = false;
  bool _isCapturing = false;
  String? _cameraError;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras == null || _cameras!.isEmpty) {
        if (mounted) {
          setState(() {
            _cameraError = 'No camera found on this device.';
            _isInitialized = true;
          });
        }
        return;
      }

      // Use rear camera at highest resolution
      final camera = _cameras!.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => _cameras!.first,
      );

      _controller = CameraController(
        camera,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _controller!.initialize();
      if (mounted) setState(() => _isInitialized = true);
    } catch (e) {
      if (mounted) {
        setState(() {
          _cameraError = 'Camera unavailable: $e';
          _isInitialized = true;
        });
      }
    }
  }

  Future<void> _captureImage() async {
    if (_controller == null || _isCapturing) return;

    setState(() => _isCapturing = true);

    try {
      final file = await _controller!.takePicture();
      final bytes = await file.readAsBytes();

      if (mounted) {
        // Navigate to screening with captured image
        Navigator.pushNamed(
          context,
          '/screening',
          arguments: bytes,
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Capture failed: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isCapturing = false);
    }
  }

  Future<void> _uploadImage() async {
    if (_isCapturing) return;
    setState(() => _isCapturing = true);

    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.image,
        withData: true,
      );
      if (!mounted || result == null || result.files.isEmpty) return;

      final pickedFile = result.files.first;
      Uint8List? bytes = pickedFile.bytes;
      if (bytes == null && pickedFile.path != null) {
        bytes = await File(pickedFile.path!).readAsBytes();
      }
      if (bytes == null || bytes.isEmpty) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Unable to read selected image')),
          );
        }
        return;
      }

      if (mounted) {
        Navigator.pushNamed(
          context,
          '/screening',
          arguments: bytes,
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Image upload failed: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isCapturing = false);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: const Text('Capture Fundus Image'),
      ),
      body: !_isInitialized
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(color: Colors.white),
                  SizedBox(height: 16),
                  Text('Initializing camera...',
                      style: TextStyle(color: Colors.white)),
                ],
              ),
            )
          : _cameraError != null
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(Icons.camera_alt_outlined,
                            size: 64, color: Colors.white70),
                        const SizedBox(height: 16),
                        Text(
                          _cameraError!,
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                              color: Colors.white, fontSize: 14),
                        ),
                        const SizedBox(height: 24),
                        OutlinedButton.icon(
                          onPressed: _isCapturing ? null : _uploadImage,
                          icon: const Icon(Icons.upload_file),
                          label: const Text('Upload Image'),
                          style: OutlinedButton.styleFrom(
                            foregroundColor: Colors.white,
                            side: const BorderSide(color: Colors.white70),
                          ),
                        ),
                      ],
                    ),
                  ),
                )
              : Stack(
                  alignment: Alignment.center,
                  children: [
                    // Camera preview
                    SizedBox.expand(
                      child: CameraPreview(_controller!),
                    ),

                    // Circle overlay guide
                    IgnorePointer(
                      child: CustomPaint(
                        size: Size.infinite,
                        painter: _FundusOverlayPainter(),
                      ),
                    ),

                    // Instructions
                    Positioned(
                      top: 16,
                      left: 16,
                      right: 16,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 16, vertical: 8),
                        decoration: BoxDecoration(
                          color: Colors.black54,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: const Text(
                          'Position the fundus within the circle.\nHold steady and tap capture.',
                          style: TextStyle(color: Colors.white, fontSize: 14),
                          textAlign: TextAlign.center,
                        ),
                      ),
                    ),

                    // Capture button
                    Positioned(
                      bottom: 40,
                      left: 16,
                      right: 16,
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          OutlinedButton.icon(
                            onPressed: _isCapturing ? null : _uploadImage,
                            icon: const Icon(Icons.upload_file),
                            label: const Text('Upload'),
                            style: OutlinedButton.styleFrom(
                              foregroundColor: Colors.white,
                              side: const BorderSide(color: Colors.white70),
                            ),
                          ),
                          const SizedBox(width: 24),
                          GestureDetector(
                            onTap: _captureImage,
                            child: Container(
                              width: 80,
                              height: 80,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                border:
                                    Border.all(color: Colors.white, width: 4),
                                color: _isCapturing
                                    ? Colors.grey
                                    : Colors.white.withValues(alpha: 0.3),
                              ),
                              child: _isCapturing
                                  ? const Padding(
                                      padding: EdgeInsets.all(20),
                                      child: CircularProgressIndicator(
                                          color: Colors.white, strokeWidth: 3),
                                    )
                                  : const Icon(Icons.camera,
                                      color: Colors.white, size: 40),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
    );
  }
}

/// Draws a semi-transparent overlay with a clear circle for fundus positioning.
class _FundusOverlayPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.shortestSide * 0.38;

    // Dark overlay outside the circle
    final path = Path()
      ..addRect(Rect.fromLTWH(0, 0, size.width, size.height))
      ..addOval(Rect.fromCircle(center: center, radius: radius))
      ..fillType = PathFillType.evenOdd;

    canvas.drawPath(
      path,
      Paint()..color = Colors.black.withValues(alpha: 0.5),
    );

    // Circle border
    canvas.drawCircle(
      center,
      radius,
      Paint()
        ..color = Colors.white.withValues(alpha: 0.7)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );

    // Corner marks
    final markLength = 20.0;
    final markPaint = Paint()
      ..color = Colors.greenAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    // Top-left of circle
    final tl = Offset(center.dx - radius, center.dy - radius);
    canvas.drawLine(tl, Offset(tl.dx + markLength, tl.dy), markPaint);
    canvas.drawLine(tl, Offset(tl.dx, tl.dy + markLength), markPaint);

    // Top-right
    final tr = Offset(center.dx + radius, center.dy - radius);
    canvas.drawLine(tr, Offset(tr.dx - markLength, tr.dy), markPaint);
    canvas.drawLine(tr, Offset(tr.dx, tr.dy + markLength), markPaint);

    // Bottom-left
    final bl = Offset(center.dx - radius, center.dy + radius);
    canvas.drawLine(bl, Offset(bl.dx + markLength, bl.dy), markPaint);
    canvas.drawLine(bl, Offset(bl.dx, bl.dy - markLength), markPaint);

    // Bottom-right
    final br = Offset(center.dx + radius, center.dy + radius);
    canvas.drawLine(br, Offset(br.dx - markLength, br.dy), markPaint);
    canvas.drawLine(br, Offset(br.dx, br.dy - markLength), markPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
