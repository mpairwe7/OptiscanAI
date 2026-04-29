"""Agentic orchestration layer for RetinalAI clinical screening platform.

Three autonomous agents manage the clinical screening lifecycle:

- ScreeningAgent: Orchestrates the full scan analysis pipeline
- MonitorAgent: Continuously watches for drift, SLA violations, and retraining triggers
- GovernanceAgent: Enforces compliance, manages review queues, generates audit reports

Agents communicate through an in-process event bus and share state via
the existing singleton services (model_service, prediction_logger, audit_trail).
"""
