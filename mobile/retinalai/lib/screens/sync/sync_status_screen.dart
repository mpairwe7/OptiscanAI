import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../providers/providers.dart';
import '../../services/sync_service.dart';

/// Sync status screen — shows pending items, sync history, manual trigger.
class SyncStatusScreen extends ConsumerStatefulWidget {
  const SyncStatusScreen({super.key});

  @override
  ConsumerState<SyncStatusScreen> createState() => _SyncStatusScreenState();
}

class _SyncStatusScreenState extends ConsumerState<SyncStatusScreen> {
  bool _isSyncing = false;
  SyncResult? _lastResult;

  Future<void> _triggerSync() async {
    setState(() => _isSyncing = true);
    try {
      final sync = await ref.read(syncProvider.future);
      final result = await sync.syncPending();
      setState(() => _lastResult = result);
      ref.invalidate(syncStatusProvider);
    } finally {
      setState(() => _isSyncing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final status = ref.watch(syncStatusProvider);
    final isOnline = ref.watch(isOnlineProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Sync Status')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Connection status
            Card(
              child: ListTile(
                leading: Icon(
                  isOnline ? Icons.cloud_done : Icons.cloud_off,
                  color: isOnline ? Colors.green : Colors.orange,
                ),
                title: Text(isOnline ? 'Online' : 'Offline'),
                subtitle: Text(isOnline
                    ? 'Connected to server'
                    : 'Working locally — predictions will sync when online'),
              ),
            ),
            const SizedBox(height: 12),

            // Pending items
            status.when(
              data: (s) => Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    children: [
                      _StatusRow('Pending sync items', '${s.pendingSyncItems}'),
                      _StatusRow('Unsynced predictions', '${s.unsyncedPredictions}'),
                      _StatusRow(
                        'Last sync',
                        s.lastSyncTime != null
                            ? _formatTime(s.lastSyncTime!)
                            : 'Never',
                      ),
                    ],
                  ),
                ),
              ),
              loading: () => const Center(child: CircularProgressIndicator()),
              error: (e, _) => Text('Error: $e'),
            ),
            const SizedBox(height: 16),

            // Sync button
            FilledButton.icon(
              onPressed: isOnline && !_isSyncing ? _triggerSync : null,
              icon: _isSyncing
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.white))
                  : const Icon(Icons.sync),
              label: Text(_isSyncing ? 'Syncing...' : 'Sync Now'),
            ),

            // Last result
            if (_lastResult != null) ...[
              const SizedBox(height: 16),
              Card(
                color: _lastResult!.status == SyncStatus.completed
                    ? Colors.green.shade50
                    : Colors.red.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Text(
                    'Synced: ${_lastResult!.synced}, '
                    'Failed: ${_lastResult!.failed}, '
                    'Remaining: ${_lastResult!.pendingRemaining}',
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  String _formatTime(DateTime dt) {
    final diff = DateTime.now().difference(dt);
    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inHours < 1) return '${diff.inMinutes}m ago';
    if (diff.inDays < 1) return '${diff.inHours}h ago';
    return '${diff.inDays}d ago';
  }
}

class _StatusRow extends StatelessWidget {
  final String label;
  final String value;

  const _StatusRow(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label),
          Text(value, style: const TextStyle(fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}
