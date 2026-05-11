import 'package:flutter/material.dart';
import 'screens/splash/splash_screen.dart';
import 'screens/home/home_screen.dart';
import 'screens/capture/camera_screen.dart';
import 'screens/screening/screening_screen.dart';
import 'screens/sync/sync_status_screen.dart';
import 'screens/settings/settings_screen.dart';

class RetinalAIApp extends StatelessWidget {
  const RetinalAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RetinalAI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF1B6B4A),
        useMaterial3: true,
        brightness: Brightness.light,
        fontFamily: 'Roboto',
      ),
      darkTheme: ThemeData(
        colorSchemeSeed: const Color(0xFF1B6B4A),
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      initialRoute: '/splash',
      routes: {
        '/splash': (context) => const SplashScreen(),
        '/home': (context) => const HomeScreen(),
        '/capture': (context) => const CameraScreen(),
        '/screening': (context) => const ScreeningScreen(),
        '/sync': (context) => const SyncStatusScreen(),
        '/settings': (context) => const SettingsScreen(),
      },
    );
  }
}
