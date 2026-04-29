"""
ClinicalKnowledgeGraph — Uganda-specific retinal disease knowledge base.

Extracted from vignn.py for reuse across all model architectures.
Provides clinical reasoning, referral priority, composite risk scoring,
treatment recommendations, and disease co-occurrence modeling.

Literature-grounded with peer-reviewed sources for each relationship.
"""

from src.models.vignn import ClinicalKnowledgeGraph, create_knowledge_graph

__all__ = ["ClinicalKnowledgeGraph", "create_knowledge_graph"]
