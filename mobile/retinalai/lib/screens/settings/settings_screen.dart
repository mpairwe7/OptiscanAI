import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../providers/providers.dart';

/// Settings screen — server URL, language, model info, debug tools.
class SettingsScreen extends ConsumerWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final isOnline = ref.watch(isOnlineProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        children: [
          // Connection
          _SectionHeader('Connection'),
          ListTile(
            leading: const Icon(Icons.dns),
            title: const Text('Server URL'),
            subtitle: const Text('http://localhost:8080'),
            trailing: Icon(
              isOnline ? Icons.check_circle : Icons.cancel,
              color: isOnline ? Colors.green : Colors.red,
            ),
          ),
          const Divider(),

          // Language
          _SectionHeader('Language'),
          ListTile(
            leading: const Icon(Icons.language),
            title: const Text('Interface Language'),
            subtitle: const Text('English'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              // TODO: Phase 2 — Luganda language selection
            },
          ),
          const Divider(),

          // Model info
          _SectionHeader('Model'),
          const ListTile(
            leading: Icon(Icons.memory),
            title: Text('Student Model'),
            subtitle: Text('MobileNetV3-Large INT8 v1.0.0'),
          ),
          const ListTile(
            leading: Icon(Icons.security),
            title: Text('Fundus Gate'),
            subtitle: Text('MobileNetV3-Small V2 Fusion'),
          ),
          const ListTile(
            leading: Icon(Icons.storage),
            title: Text('Bundle Version'),
            subtitle: Text('1.0.0'),
          ),
          const Divider(),

          // Audit
          _SectionHeader('Audit & Privacy'),
          ListTile(
            leading: const Icon(Icons.verified_user),
            title: const Text('Verify Audit Chain'),
            onTap: () async {
              final audit = await ref.read(auditProvider.future);
              final result = await audit.verifyChain();
              if (context.mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text(result.valid
                        ? 'Audit chain valid (${result.entriesChecked} entries)'
                        : 'Chain broken at ${result.brokenAtId}'),
                  ),
                );
              }
            },
          ),
          const Divider(),

          // About
          _SectionHeader('About'),
          const ListTile(
            leading: Icon(Icons.info),
            title: Text('RetinalAI Clinical Screening'),
            subtitle: Text('Version 1.0.0\nOffline-first retinal disease detection'),
          ),
        ],
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  final String title;
  const _SectionHeader(this.title);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
      child: Text(
        title,
        style: Theme.of(context).textTheme.titleSmall?.copyWith(
              color: Theme.of(context).colorScheme.primary,
            ),
      ),
    );
  }
}
