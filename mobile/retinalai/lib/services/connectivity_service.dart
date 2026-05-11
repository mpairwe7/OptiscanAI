import 'dart:async';

import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:http/http.dart' as http;

/// Network connectivity detection + server reachability.
///
/// Provides both raw connectivity state (wifi/mobile/none) and actual
/// server reachability via a lightweight ping. This matters in rural Uganda
/// where a device may report "mobile connected" but have no actual data
/// throughput on congested 2G towers.
class ConnectivityService {
  final Connectivity _connectivity = Connectivity();
  final String _serverBaseUrl;
  final Duration _pingTimeout;

  StreamController<ConnectivityState>? _controller;
  StreamSubscription<List<ConnectivityResult>>? _subscription;
  ConnectivityState _lastState = ConnectivityState.unknown;

  ConnectivityService({
    String serverBaseUrl = 'http://localhost:8080',
    Duration pingTimeout = const Duration(seconds: 5),
  })  : _serverBaseUrl = serverBaseUrl,
        _pingTimeout = pingTimeout;

  /// Stream of connectivity state changes.
  Stream<ConnectivityState> get stateStream {
    _controller ??= StreamController<ConnectivityState>.broadcast(
      onListen: _startListening,
      onCancel: _stopListening,
    );
    return _controller!.stream;
  }

  ConnectivityState get currentState => _lastState;
  bool get isOnline => _lastState == ConnectivityState.online;

  void _startListening() {
    _subscription = _connectivity.onConnectivityChanged.listen((results) async {
      final result = results.isNotEmpty ? results.first : ConnectivityResult.none;
      if (result == ConnectivityResult.none) {
        _updateState(ConnectivityState.offline);
      } else {
        // Device has connectivity — check if server is actually reachable
        final reachable = await canReachServer();
        _updateState(
            reachable ? ConnectivityState.online : ConnectivityState.limited);
      }
    });

    // Initial check
    _checkNow();
  }

  void _stopListening() {
    _subscription?.cancel();
    _subscription = null;
  }

  void _updateState(ConnectivityState state) {
    if (state != _lastState) {
      _lastState = state;
      _controller?.add(state);
    }
  }

  Future<void> _checkNow() async {
    final results = await _connectivity.checkConnectivity();
    final result = results.isNotEmpty ? results.first : ConnectivityResult.none;
    if (result == ConnectivityResult.none) {
      _updateState(ConnectivityState.offline);
    } else {
      final reachable = await canReachServer();
      _updateState(
          reachable ? ConnectivityState.online : ConnectivityState.limited);
    }
  }

  /// Ping the server's health endpoint to confirm actual reachability.
  Future<bool> canReachServer() async {
    try {
      final response = await http
          .get(Uri.parse('$_serverBaseUrl/health'))
          .timeout(_pingTimeout);
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  void dispose() {
    _stopListening();
    _controller?.close();
    _controller = null;
  }
}

enum ConnectivityState {
  /// Server is reachable and responding.
  online,

  /// Device has network but server is unreachable.
  limited,

  /// No network connectivity at all.
  offline,

  /// Not yet determined.
  unknown,
}
