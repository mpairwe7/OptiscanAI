import 'package:flutter/material.dart';

/// Persistent banner shown when the app is in offline mode.
class OfflineBanner extends StatelessWidget {
  const OfflineBanner({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 16),
      color: Colors.orange.shade700,
      child: const Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.wifi_off, color: Colors.white, size: 16),
          SizedBox(width: 8),
          Text(
            'Offline — screening results stored locally',
            style: TextStyle(color: Colors.white, fontSize: 13),
          ),
        ],
      ),
    );
  }
}
