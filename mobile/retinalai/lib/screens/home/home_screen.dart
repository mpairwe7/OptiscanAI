import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../providers/providers.dart';
import '../../services/sync_service.dart';
import '../../widgets/offline_banner.dart';

/// Home screen — dashboard with screening history and quick actions.
class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final isOnline = ref.watch(isOnlineProvider);
    final syncStatus = ref.watch(syncStatusProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('RetinalAI'),
        actions: [
          IconButton(
            icon: const Icon(Icons.sync),
            onPressed: () => Navigator.pushNamed(context, '/sync'),
          ),
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () => Navigator.pushNamed(context, '/settings'),
          ),
        ],
      ),
      body: Column(
        children: [
          if (!isOnline) const OfflineBanner(),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Quick stats
                  _StatsCard(syncStatus: syncStatus),
                  const SizedBox(height: 16),

                  // Start screening button
                  SizedBox(
                    height: 72,
                    child: FilledButton.icon(
                      onPressed: () => Navigator.pushNamed(context, '/capture'),
                      icon: const Icon(Icons.camera_alt, size: 28),
                      label: Text(
                        'Start Screening',
                        style: theme.textTheme.titleMedium?.copyWith(
                          color: theme.colorScheme.onPrimary,
                        ),
                      ),
                      style: FilledButton.styleFrom(
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(16),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),

                  // Recent screenings
                  Text(
                    'Recent Screenings',
                    style: theme.textTheme.titleMedium,
                  ),
                  const SizedBox(height: 8),
                  Expanded(child: _RecentScreeningsList()),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _StatsCard extends StatelessWidget {
  final AsyncValue<SyncStatusSummary> syncStatus;

  const _StatsCard({required this.syncStatus});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: syncStatus.when(
          data: (status) => Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _StatItem(
                icon: Icons.cloud_off,
                label: 'Pending Sync',
                value: '${status.pendingSyncItems}',
                color: status.pendingSyncItems > 0
                    ? theme.colorScheme.error
                    : theme.colorScheme.primary,
              ),
              _StatItem(
                icon: Icons.visibility,
                label: 'Unsynced',
                value: '${status.unsyncedPredictions}',
                color: theme.colorScheme.tertiary,
              ),
              _StatItem(
                icon: status.isOnline ? Icons.wifi : Icons.wifi_off,
                label: status.isOnline ? 'Online' : 'Offline',
                value: status.isOnline ? 'Connected' : 'Local',
                color: status.isOnline
                    ? theme.colorScheme.primary
                    : theme.colorScheme.outline,
              ),
            ],
          ),
          loading: () => const Center(child: CircularProgressIndicator()),
          error: (_, __) => const Text('Failed to load status'),
        ),
      ),
    );
  }
}

class _StatItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final Color color;

  const _StatItem({
    required this.icon,
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Icon(icon, color: color, size: 24),
        const SizedBox(height: 4),
        Text(value,
            style: Theme.of(context)
                .textTheme
                .titleSmall
                ?.copyWith(color: color, fontWeight: FontWeight.bold)),
        Text(label, style: Theme.of(context).textTheme.bodySmall),
      ],
    );
  }
}

class _RecentScreeningsList extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final db = ref.watch(databaseProvider);

    return db.when(
      data: (database) => StreamBuilder(
        stream: database.watchRecentPredictions(limit: 20),
        builder: (context, snapshot) {
          if (!snapshot.hasData || snapshot.data!.isEmpty) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.remove_red_eye_outlined,
                      size: 64,
                      color: Theme.of(context).colorScheme.outlineVariant),
                  const SizedBox(height: 16),
                  Text('No screenings yet',
                      style: Theme.of(context).textTheme.bodyLarge),
                  const SizedBox(height: 8),
                  Text('Tap "Start Screening" to begin',
                      style: Theme.of(context).textTheme.bodySmall),
                ],
              ),
            );
          }

          final predictions = snapshot.data!;
          return ListView.builder(
            itemCount: predictions.length,
            itemBuilder: (context, index) {
              final p = predictions[index];
              return ListTile(
                leading: Icon(
                  p.gatePassed ? Icons.check_circle : Icons.cancel,
                  color: p.gatePassed ? Colors.green : Colors.red,
                ),
                title: Text(p.detectedDiseases.isNotEmpty
                    ? p.detectedDiseases
                    : 'No diseases detected'),
                subtitle: Text(
                  '${p.inferenceSource} | ${p.inferenceMs?.toStringAsFixed(0) ?? "?"}ms',
                ),
                trailing: Icon(
                  p.synced ? Icons.cloud_done : Icons.cloud_off,
                  size: 18,
                  color: p.synced ? Colors.green : Colors.orange,
                ),
              );
            },
          );
        },
      ),
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (e, _) => Center(child: Text('Error: $e')),
    );
  }
}
