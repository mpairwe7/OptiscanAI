#!/usr/bin/env python3
"""Extract ClinicalKnowledgeGraph to compact JSON for on-device use.

Serializes disease relationships, referral rules, co-occurrence patterns,
treatment recommendations, and Uganda prevalence data from the Python
ClinicalKnowledgeGraph into a JSON format consumable by the Flutter app.

Usage:
    PYTHONPATH=. python scripts/extract_clinical_kg_json.py \
        --output outputs/mobile_export/clinical_kg.json

Produces:
    clinical_kg.json (~0.5 MB) with:
        - diseases: per-disease metadata
        - co_occurrence: disease relationship edges
        - referral_rules: priority thresholds and categories
        - uganda_prevalence: disease prevalence data
        - disease_categories: groupings
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Standard RFMiD disease names (28 classes after filtering)
DISEASE_NAMES = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS",
    "MS", "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST",
    "AION", "PT", "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS",
]


def extract_kg(disease_names: list[str]) -> dict:
    """Extract full knowledge graph to serializable dict."""
    from src.models.vignn import ClinicalKnowledgeGraph

    kg = ClinicalKnowledgeGraph(disease_names=disease_names)

    cooccurrence_map = getattr(kg, "cooccurrence", {})
    treatment_map = getattr(kg, "treatment_considerations", {})
    prevalence_map = getattr(kg, "uganda_prevalence", {})
    categories_map = getattr(kg, "categories", {})

    # --- Diseases ---
    diseases = {}
    for name in disease_names:
        info = kg.get_disease_info(name) if hasattr(kg, "get_disease_info") else {}
        diseases[name] = {
            "code": name,
            "related_diseases": cooccurrence_map.get(name, []),
            "treatment": treatment_map.get(name, []),
            "uganda_prevalence": prevalence_map.get(name),
        }

    # --- Co-occurrence edges ---
    co_occurrence = []
    seen = set()
    for disease, related in cooccurrence_map.items():
        for rel in related:
            edge_key = tuple(sorted([disease, rel]))
            if edge_key not in seen:
                seen.add(edge_key)
                co_occurrence.append({
                    "from": disease,
                    "to": rel,
                })

    # --- Disease categories ---
    categories = {}
    for cat_name, members in categories_map.items():
        resolved = [m for m in members if m in disease_names]
        if resolved:
            categories[cat_name] = resolved

    # --- Referral rules ---
    referral_rules = {
        "priority_levels": {
            "EMERGENCY": {
                "score": 5,
                "description": "Sight-threatening, immediate referral",
                "max_hours": 0,
            },
            "URGENT": {
                "score": 4,
                "description": "Refer within 24 hours",
                "max_hours": 24,
            },
            "ROUTINE": {
                "score": 3,
                "description": "Refer within 1 week",
                "max_hours": 168,
            },
            "FOLLOW_UP": {
                "score": 2,
                "description": "Schedule routine follow-up",
                "max_hours": 720,
            },
            "NORMAL": {
                "score": 1,
                "description": "No pathology detected",
                "max_hours": None,
            },
        },
        "emergency_diseases": [],
        "urgent_diseases": [],
    }

    # Classify diseases by referral urgency using the KG
    for disease in disease_names:
        priority = kg.get_referral_priority([disease]) if hasattr(kg, "get_referral_priority") else "FOLLOW_UP"
        if priority == "EMERGENCY":
            referral_rules["emergency_diseases"].append(disease)
        elif priority == "URGENT":
            referral_rules["urgent_diseases"].append(disease)

    # --- Uganda prevalence ---
    uganda_prevalence = {}
    for disease in disease_names:
        prev = prevalence_map.get(disease)
        if prev is not None:
            uganda_prevalence[disease] = prev

    # --- Alias map ---
    alias_map = {}
    if hasattr(kg, "alias_map"):
        alias_map = {k: v for k, v in kg.alias_map.items() if v is not None}

    result = {
        "version": "1.0.0",
        "num_diseases": len(disease_names),
        "disease_names": disease_names,
        "diseases": diseases,
        "co_occurrence": co_occurrence,
        "disease_categories": categories,
        "referral_rules": referral_rules,
        "uganda_prevalence": uganda_prevalence,
        "alias_map": alias_map,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract Clinical KG to JSON")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/mobile_export/clinical_kg.json",
    )
    parser.add_argument(
        "--disease-names",
        type=str,
        nargs="+",
        default=None,
        help="Override disease names list",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    disease_names = args.disease_names or DISEASE_NAMES
    logger.info("Extracting KG for %d diseases", len(disease_names))

    kg_data = extract_kg(disease_names)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(kg_data, f, indent=2, ensure_ascii=False)

    size_kb = output_path.stat().st_size / 1024
    logger.info(
        "Clinical KG exported: %s (%.1f KB, %d diseases, %d edges)",
        output_path,
        size_kb,
        len(kg_data["diseases"]),
        len(kg_data["co_occurrence"]),
    )
    print(f"Exported clinical KG: {output_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
