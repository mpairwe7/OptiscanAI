"""Compact clinical narrator (vocabulary-pruned, prose-only)."""

from .compact import CompactNarrator, findings_summary, narrative_prompt

__all__ = ["CompactNarrator", "findings_summary", "narrative_prompt"]
