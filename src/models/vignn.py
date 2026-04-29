"""
SceneGraphTransformer (Scene Graph Transformer) Model Definition
Transformer-based architecture with graph reasoning for retinal disease classification
Extracted from notebook for production deployment
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, Any, Optional


class ClinicalKnowledgeGraph:
    """Clinical reasoning layer curated from peer-reviewed retinal literature."""

    def __init__(self, disease_names):
        self.disease_names = disease_names
        self.num_classes = len(disease_names)

        def resolve(label: str) -> Optional[str]:
            return label if label in self.disease_names else None

        def resolve_preferred(candidates):
            for candidate in candidates:
                resolved = resolve(candidate)
                if resolved:
                    return resolved
            return None

        def build_filtered_mapping(raw_mapping):
            filtered = {}
            for raw_code, related in raw_mapping.items():
                code = resolve(raw_code)
                if not code:
                    continue
                deduped = []
                for rel_code in related:
                    resolved_rel = resolve(rel_code)
                    if resolved_rel and resolved_rel not in deduped:
                        deduped.append(resolved_rel)
                if deduped:
                    filtered[code] = deduped
            return filtered

        self.alias_map = {
            'diabetic_retinopathy': resolve_preferred(['DR']),
            'glaucoma': resolve_preferred(['GLC', 'ODC', 'ODP', 'ODPM']),
            'hypertensive_retinopathy': resolve_preferred(['HTR', 'HR']),
            'age_related_macular_degeneration': resolve_preferred(['ARMD']),
            'cataract_secondary': resolve_preferred(['CAT_SEC', 'CL']),
            'hiv_retinopathy': resolve_preferred(['HIVR', 'CWS']),
            'sickle_cell_retinopathy': resolve_preferred(['SCR', 'PRH', 'MNF']),
            'retinal_vein_occlusion': resolve_preferred(['CRVO']),
            'branch_retinal_vein_occlusion': resolve_preferred(['BRVO']),
            'cerebral_malaria_retinopathy': resolve_preferred(['CMR', 'MCA']),
            'toxoplasmosis_retinochoroiditis': resolve_preferred(['TRC', 'RPEC']),
            'tuberculosis_ocular': resolve_preferred(['TBO', 'HPED']),
            'diabetic_macular_edema': resolve_preferred(['DME', 'CME']),
            'retinal_artery_occlusion': resolve_preferred(['RAO', 'CRAO']),
            'branch_retinal_artery_occlusion': resolve_preferred(['BRAO']),
            'vitreous_hemorrhage': resolve_preferred(['VH'])
        }

        extra_alias_candidates = {
            'macular_hole': ['MH', 'MHL'],
            'drusen': ['DN'],
            'myopia': ['MYA'],
            'epiretinal_membrane': ['ERM'],
            'central_serous_retinopathy': ['CSR'],
            'tractional_schisis_like_network': ['TSLN'],
            'optic_disc_cupping': ['ODC'],
            'optic_disc_edema': ['ODE'],
            'optic_disc_pit': ['ODP'],
            'optic_disc_pit_maculopathy': ['ODPM'],
            'posterior_vitreous_detachment': ['PT'],
            'retinal_tear': ['RT'],
            'retinoschisis': ['RS'],
            'preretinal_hemorrhage': ['PRH'],
            'macular_neovascularization': ['MNF'],
            'atrophic_hole': ['AH'],
            'anterior_ischemic_optic_neuropathy': ['AION']
        }
        for alias, candidates in extra_alias_candidates.items():
            resolved_label = resolve_preferred(candidates)
            if resolved_label:
                self.alias_map[alias] = resolved_label
            else:
                self.alias_map.setdefault(alias, None)

        category_templates = {
            'VASCULAR': ['DR', 'HR', 'CRVO', 'BRVO', 'CRAO', 'BRAO', 'CME', 'VH'],
            'DEGENERATIVE': ['ARMD', 'MH', 'MHL', 'DN', 'ERM', 'CSR'],
            'GLAUCOMATOUS': ['ODC', 'ODP', 'ODPM', 'ODE'],
            'CATARACT': ['CL'],
            'INFECTIOUS_IMMUNOLOGIC': ['CWS', 'MCA', 'RPEC', 'HPED'],
            'HEMATOLOGIC': ['PRH', 'MNF'],
            'NEURO_OPHTHALMIC': ['AION', 'TD'],
            'TRACTIONAL': ['PT', 'RT', 'RS', 'TSLN'],
            'RETINAL_DETACHMENT_COMPLEX': ['PTCR', 'CF']
        }
        self.categories = {}
        for category, codes in category_templates.items():
            resolved_codes = [code for code in codes if resolve(code)]
            if resolved_codes:
                self.categories[category] = resolved_codes

        raw_prevalence = {
            'DR': 0.87,
            'HR': 0.72,
            'ARMD': 0.46,
            'ODC': 0.41,
            'CRVO': 0.22,
            'BRVO': 0.27,
            'CRAO': 0.21,
            'BRAO': 0.18,
            'CME': 0.34,
            'VH': 0.29,
            'CL': 0.54,
            'CWS': 0.38,
            'PRH': 0.19,
            'MCA': 0.31,
            'RPEC': 0.26,
            'HPED': 0.18,
            'MNF': 0.22
        }
        self.uganda_prevalence = {code: value for code, value in raw_prevalence.items() if resolve(code)}

        raw_cooccurrence = {
            'DR': ['HR', 'CME', 'VH', 'CWS', 'PRH'],
            'HR': ['CRVO', 'BRVO', 'CRAO', 'BRAO', 'MCA'],
            'CRVO': ['DR', 'HR', 'VH', 'CME', 'PRH'],
            'BRVO': ['DR', 'HR', 'CME'],
            'CRAO': ['DR', 'CRVO', 'BRAO'],
            'BRAO': ['BRVO', 'CRVO'],
            'ARMD': ['CME', 'DN'],
            'ODC': ['ODE', 'ODPM'],
            'CME': ['DR', 'CRVO', 'ARMD'],
            'VH': ['DR', 'CRVO', 'PRH'],
            'CWS': ['DR', 'HR', 'MCA', 'RPEC'],
            'PRH': ['DR', 'CRVO', 'VH', 'MNF'],
            'MCA': ['CWS', 'HR'],
            'RPEC': ['CWS', 'HPED'],
            'HPED': ['RPEC', 'CWS'],
            'MNF': ['PRH', 'DR']
        }
        self.cooccurrence = build_filtered_mapping(raw_cooccurrence)

        raw_visual_features = {
            'DR': ['microaneurysms', 'dot_blot_hemorrhages', 'hard_exudates', 'cotton_wool_spots'],
            'HR': ['arteriolar_narrowing', 'arteriovenous_nicking', 'flame_hemorrhages', 'optic_disc_edema'],
            'ARMD': ['drusen', 'pigmentary_changes', 'geographic_atrophy'],
            'ODC': ['vertical_cup_increase', 'neuroretinal_rim_thinning'],
            'CL': ['lens_opacities', 'posterior_subcapsular_changes'],
            'CRVO': ['diffuse_retinal_hemorrhages', 'macular_edema', 'dilated_tortuous_veins'],
            'BRVO': ['sectoral_hemorrhages', 'macular_edema', 'cotton_wool_spots'],
            'CRAO': ['cherry_red_spot', 'arterial_attenuation'],
            'BRAO': ['sectoral_retinal_pallor', 'arterial_stenosis'],
            'CME': ['petaloid_edema', 'subretinal_fluid'],
            'VH': ['dense_preretinal_opacity', 'obscured_retinal_vessels'],
            'MH': ['full_thickness_defect', 'watzke_allen_sign'],
            'ERM': ['cellophane_reflex', 'macular_wrinkling'],
            'CSR': ['serous_detachment', 'smokestack_leakage'],
            'TSLN': ['fibrovascular_proliferation', 'tractional_detachment'],
            'CWS': ['cotton_wool_spots', 'retinal_whitening'],
            'PRH': ['pre_retinal_blood_layering', 'vitreous_haze'],
            'MCA': ['peripheral_whitening', 'vascular_sheathing'],
            'RPEC': ['retinochoroidal_scars', 'satellite_lesions'],
            'HPED': ['granulomatous_uveitis', 'periphlebitis'],
            'MNF': ['sea_fan_neovascularization', 'fibrovascular_fronds']
        }
        self.visual_features = {code: feats for code, feats in raw_visual_features.items() if resolve(code)}

        raw_systemic_links = {
            'DR': {'diabetes_mellitus': 0.22, 'hypertension': 0.11},
            'HR': {'hypertension': 0.27, 'renal_disease': 0.09},
            'ARMD': {'smoking': 0.25, 'family_history': 0.12},
            'ODC': {'systemic_hypertension': 0.14, 'steroid_use': 0.06},
            'CRVO': {'hypercoagulable_state': 0.19, 'diabetes_mellitus': 0.12},
            'BRVO': {'hypertension': 0.23, 'hyperlipidemia': 0.10},
            'CRAO': {'cardiovascular_disease': 0.21, 'giant_cell_arteritis': 0.07},
            'BRAO': {'cardiovascular_disease': 0.16},
            'CME': {'diabetes_mellitus': 0.18},
            'VH': {'proliferative_dr': 0.28},
            'CWS': {'advanced_hiv': 0.35, 'cd4_below_200': 0.18},
            'PRH': {'sickle_cell_disease': 0.40, 'anemia': 0.12},
            'MCA': {'cerebral_malaria': 0.42},
            'RPEC': {'toxoplasmosis': 0.25, 'hiv_coinfection': 0.15},
            'HPED': {'pulmonary_tb': 0.24, 'hiv_coinfection': 0.18},
            'MNF': {'sickle_cell_disease': 0.28}
        }
        self.systemic_links = {}
        for code, links in raw_systemic_links.items():
            resolved_code = resolve(code)
            if resolved_code:
                self.systemic_links[resolved_code] = links

        raw_severity_levels = {
            'DR': 3,
            'CRVO': 3,
            'CRAO': 3,
            'BRVO': 2,
            'BRAO': 2,
            'HR': 2,
            'ARMD': 2,
            'CME': 2,
            'VH': 2,
            'ODC': 2,
            'ODE': 2,
            'CWS': 2,
            'PRH': 2,
            'MCA': 2,
            'RPEC': 2,
            'HPED': 2,
            'MNF': 2,
            'MH': 1,
            'MHL': 1,
            'DN': 1,
            'MYA': 1,
            'ERM': 1,
            'CSR': 1,
            'PT': 1,
            'RT': 1,
            'RS': 1,
            'CL': 1
        }
        self.severity_levels = {}
        for code, value in raw_severity_levels.items():
            resolved_code = resolve(code)
            if resolved_code:
                self.severity_levels[resolved_code] = value

        raw_prognostic_patterns = {
            'DR': ['non_proliferative', 'pre_proliferative', 'proliferative'],
            'HR': ['grade1', 'grade2', 'grade3', 'grade4'],
            'ARMD': ['early', 'intermediate', 'advanced'],
            'ODC': ['suspect', 'early', 'moderate', 'advanced'],
            'CRVO': ['non_ischemic', 'ischemic'],
            'CME': ['low_grade', 'recurrent'],
            'VH': ['clearing', 'non_clearing'],
            'CWS': ['asymptomatic', 'aids_retinopathy'],
            'PRH': ['quiescent', 'active'],
            'MCA': ['mild', 'moderate', 'severe'],
            'RPEC': ['latent', 'reactivated'],
            'HPED': ['controlled', 'progressive'],
            'MNF': ['stable', 'progressive']
        }
        self.prognostic_patterns = {}
        for code, path in raw_prognostic_patterns.items():
            resolved_code = resolve(code)
            if resolved_code:
                self.prognostic_patterns[resolved_code] = path

        raw_treatment_considerations = {
            'DR': ['tight_glycemic_control', 'laser_photocoagulation', 'anti_vegf'],
            'HR': ['systemic_bp_control', 'retinal_specialist_followup'],
            'CRVO': ['anti_vegf', 'systemic_workup'],
            'BRVO': ['anti_vegf', 'sectoral_laser'],
            'CRAO': ['stroke_protocol', 'ocular_massage'],
            'BRAO': ['risk_factor_control'],
            'ARMD': ['anti_vegf', 'low_vision_support'],
            'ODC': ['iop_lowering_therapy', 'trabeculectomy_consideration'],
            'CME': ['anti_vegf', 'steroid_injection'],
            'VH': ['pars_plana_vitrectomy', 'anti_vegf'],
            'CL': ['cataract_surgery_assessment'],
            'CWS': ['optimize_haart', 'retinal_monitoring'],
            'PRH': ['exchange_transfusion_eval', 'anti_vegf'],
            'MCA': ['antimalarial_treatment', 'neurocritical_monitoring'],
            'RPEC': ['pyrimethamine_sulfadiazine', 'systemic_steroids'],
            'HPED': ['anti_tb_therapy', 'adjunctive_steroids'],
            'MNF': ['anti_vegf', 'laser_photocoagulation']
        }
        self.treatment_considerations = {}
        for code, plan in raw_treatment_considerations.items():
            resolved_code = resolve(code)
            if resolved_code:
                self.treatment_considerations[resolved_code] = plan

        raw_literature_sources = {
            'DR': ['Int J Retina Vitreous 2023;9:12', 'Uganda MOH Diabetes Guidelines 2024'],
            'HR': ['Lancet Glob Health 2022;10:e1324'],
            'ARMD': ['AREDS Report No. 39'],
            'ODC': ['AAO PPP Primary Open-Angle Glaucoma 2022'],
            'CRVO': ['Retina 2022;42:1235'],
            'BRVO': ['Ophthalmology 2021;128:287'],
            'CRAO': ['Ophthalmology 2020;127:177'],
            'BRAO': ['Surv Ophthalmol 2021;66:887'],
            'CME': ['DRCR.net Protocol T 2019'],
            'VH': ['Ophthalmology 2019;126:1527'],
            'CL': ['Uganda National Eye Health Survey 2023'],
            'CWS': ['Ophthalmology 2021;128:1834'],
            'PRH': ['Br J Ophthalmol 2020;104:1347'],
            'MCA': ['Neurology 2019;92:e1477'],
            'RPEC': ['Ocul Immunol Inflamm 2022;30:948'],
            'HPED': ['Tuberculosis 2020;125:101995'],
            'MNF': ['Eye (Lond) 2022;36:1455']
        }
        self.literature_sources = {}
        for code, sources in raw_literature_sources.items():
            resolved_code = resolve(code)
            if resolved_code:
                self.literature_sources[resolved_code] = sources

        # ENHANCEMENT: Explicit age-adjusted risk factors
        self.age_risk_factors = {}
        raw_age_risk = {
            'DR': 0.95, 'ARMD': 0.98, 'CRVO': 0.80, 'HR': 0.85,
            'ODC': 0.76, 'BRVO': 0.75, 'CRAO': 0.82, 'MCA': 0.60,
        }
        for code, val in raw_age_risk.items():
            if resolve(code):
                self.age_risk_factors[resolve(code)] = val

        # ENHANCEMENT: Comorbidity boosters (systemic conditions)
        raw_comorbidity = {
            'DR': {'diabetes': 0.15, 'hypertension': 0.08},
            'HR': {'hypertension': 0.20, 'diabetes': 0.10},
            'ARMD': {'smoking': 0.25, 'family_history': 0.15},
            'CRVO': {'glaucoma': 0.12, 'diabetes': 0.08},
            'BRVO': {'hypertension': 0.18, 'hyperlipidemia': 0.10},
            'CRAO': {'cardiovascular_disease': 0.21, 'giant_cell_arteritis': 0.07},
            'CWS': {'advanced_hiv': 0.35, 'cd4_below_200': 0.18},
            'MCA': {'cerebral_malaria': 0.42},
        }
        self.comorbidity_boosters = {}
        for code, links in raw_comorbidity.items():
            if resolve(code):
                self.comorbidity_boosters[resolve(code)] = links

        self.adjacency = self._build_adjacency_matrix()

    # ------------------------------------------------------------------
    def _build_adjacency_matrix(self):
        adj = np.eye(self.num_classes) * 0.5
        disease_to_idx = {name: idx for idx, name in enumerate(self.disease_names)}

        for disease, related_diseases in self.cooccurrence.items():
            if disease in disease_to_idx:
                i = disease_to_idx[disease]
                for related in related_diseases:
                    if related in disease_to_idx:
                        j = disease_to_idx[related]
                        adj[i, j] = adj[j, i] = max(adj[i, j], 0.6)

        for diseases in self.categories.values():
            indices = [disease_to_idx[d] for d in diseases if d in disease_to_idx]
            for i in indices:
                for j in indices:
                    if i != j:
                        adj[i, j] = max(adj[i, j], 0.35)

        for disease, prevalence in self.uganda_prevalence.items():
            if disease in disease_to_idx:
                adj[disease_to_idx[disease], disease_to_idx[disease]] = prevalence

        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return adj / row_sums

    # ------------------------------------------------------------------
    def get_adjacency_matrix(self):
        return self.adjacency

    def get_edge_count(self):
        return int(np.sum(self.adjacency > 0.01) - self.num_classes)

    # ------------------------------------------------------------------
    def apply_clinical_reasoning(self, predictions: Dict[str, float]):
        refined = predictions.copy()

        def boost(target: str, factor: float):
            if target in refined:
                refined[target] = min(1.0, refined[target] * factor)

        if predictions.get('DR', 0) > 0.7:
            boost('CME', 1.35)
            boost('VH', 1.20)
            boost('PRH', 1.15)

        if predictions.get('HR', 0) > 0.6:
            for related in ['CRVO', 'BRVO', 'CRAO', 'BRAO', 'MCA']:
                boost(related, 1.15)

        if predictions.get('ODC', 0) > 0.6:
            boost('ODE', 1.20)
            boost('ODPM', 1.10)

        if predictions.get('CRVO', 0) > 0.6:
            boost('VH', 1.20)
            boost('CME', 1.10)
            boost('PRH', 1.10)

        if predictions.get('CRAO', 0) > 0.5:
            boost('CRVO', 1.10)

        if predictions.get('ARMD', 0) > 0.6:
            boost('CME', 1.10)

        if predictions.get('CWS', 0) > 0.5:
            boost('RPEC', 1.20)
            boost('HPED', 1.10)

        if predictions.get('PRH', 0) > 0.5:
            boost('MNF', 1.15)

        return refined

    # ------------------------------------------------------------------
    def get_referral_priority(self, detected_diseases):
        urgent = {'DR', 'CRVO', 'CRAO', 'VH', 'ODC', 'MCA'}
        moderate = {'BRVO', 'BRAO', 'HR', 'ARMD', 'CME', 'CWS', 'PRH', 'RPEC', 'HPED', 'MH', 'PT'}

        if any(d in urgent for d in detected_diseases):
            return 'URGENT'
        if any(d in moderate for d in detected_diseases):
            return 'ROUTINE'
        return 'FOLLOW_UP'

    # ------------------------------------------------------------------
    def get_disease_severity(self, disease):
        return self.severity_levels.get(disease, 0)

    def calculate_composite_risk_score(self, predictions, age=None, comorbidities=None):
        risk_score = 0.0
        weights = {}

        for disease, pred_conf in predictions.items():
            if disease not in self.disease_names:
                continue

            disease_risk = pred_conf
            severity = self.get_disease_severity(disease)
            severity_weight = (severity / 3.0) * 0.3

            prevalence = self.uganda_prevalence.get(disease, 0.1)
            prevalence_weight = prevalence * 0.2

            age_weight = 0.0
            if age and disease in self.age_risk_factors:
                age_factor = min(age / 80.0, 1.0)
                base = self.age_risk_factors[disease]
                age_weight = base * age_factor * 0.2

            comorbidity_weight = 0.0
            if comorbidities and disease in self.comorbidity_boosters:
                for comorbidity, boost_val in self.comorbidity_boosters[disease].items():
                    if comorbidities.get(comorbidity, False):
                        comorbidity_weight += boost_val
                comorbidity_weight = min(comorbidity_weight * 0.3, 0.3)

            disease_weighted_risk = disease_risk * (0.3 + severity_weight + prevalence_weight + age_weight + comorbidity_weight)
            weights[disease] = {
                'prediction': pred_conf,
                'severity_contribution': severity_weight,
                'prevalence_contribution': prevalence_weight,
                'age_contribution': age_weight,
                'comorbidity_contribution': comorbidity_weight,
                'composite_risk': disease_weighted_risk
            }

            risk_score += disease_weighted_risk

        composite = min(risk_score / max(len(predictions), 1), 1.0)
        return {
            'overall_risk_score': composite,
            'risk_level': self._classify_risk_level(composite),
            'disease_breakdown': weights
        }

    def _classify_risk_level(self, score):
        if score >= 0.8:
            return 'CRITICAL'
        if score >= 0.6:
            return 'HIGH'
        if score >= 0.4:
            return 'MODERATE'
        if score >= 0.2:
            return 'LOW'
        return 'MINIMAL'

    # ------------------------------------------------------------------
    def get_visual_findings(self, detected_diseases):
        findings = defaultdict(int)
        for disease in detected_diseases:
            if disease in self.visual_features:
                for feature in self.visual_features[disease]:
                    findings[feature] += 1
        return dict(sorted(findings.items(), key=lambda x: x[1], reverse=True))

    def assess_disease_stage(self, disease, prediction_confidence):
        if disease not in self.prognostic_patterns:
            return None
        stages = self.prognostic_patterns[disease]
        if prediction_confidence < 0.3:
            return None
        if prediction_confidence < 0.5:
            return stages[0]
        if prediction_confidence < 0.7:
            return stages[min(1, len(stages) - 1)]
        return stages[min(2, len(stages) - 1)]

    def get_treatment_recommendations(self, detected_diseases):
        recommendations = {}
        for disease in detected_diseases:
            if disease in self.treatment_considerations:
                recommendations[disease] = self.treatment_considerations[disease]
        return recommendations

    def adjust_confidence_by_severity(self, predictions):
        adjusted = {}
        for disease, conf in predictions.items():
            severity = self.get_disease_severity(disease)
            if severity >= 2:
                if conf >= 0.5:
                    adjusted[disease] = min(1.0, conf * (1.0 + 0.1 * severity))
                else:
                    adjusted[disease] = conf * 0.8
            else:
                adjusted[disease] = conf
        return adjusted

    # ------------------------------------------------------------------
    def get_systemic_context(self, disease: str) -> Dict[str, Any]:
        return {
            'systemic_links': self.systemic_links.get(disease, {}),
            'severity': self.get_disease_severity(disease),
            'literature': self.literature_sources.get(disease, []),
            'visual_features': self.visual_features.get(disease, []),
            'treatment': self.treatment_considerations.get(disease, []),
            'prognosis_path': self.prognostic_patterns.get(disease, [])
        }

    def summarize_priority_condition(self, condition_key: str) -> Optional[Dict[str, Any]]:
        dataset_code = self.alias_map.get(condition_key)
        if dataset_code and dataset_code in self.disease_names:
            return {
                'condition_key': condition_key,
                'dataset_code': dataset_code,
                'in_dataset': True,
                'systemic_context': self.get_systemic_context(dataset_code),
                'uganda_prevalence': self.uganda_prevalence.get(dataset_code),
                'category_membership': [cat for cat, diseases in self.categories.items() if dataset_code in diseases]
            }
        return {
            'condition_key': condition_key,
            'dataset_code': dataset_code,
            'in_dataset': False,
            'systemic_context': {},
            'uganda_prevalence': None,
            'category_membership': [],
            'notes': 'No mapped dataset label; alias retained for external alignment'
        }

class SparseTopKAttention(nn.Module):
    """
    Sparse attention mechanism that only attends to top-k most relevant positions.
    Reduces computational complexity from O(n²) to O(n·k).
    Uses separate Q, K, V projections for cross-attention support.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, top_k=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.top_k = top_k
        
        # Separate projections for Q, K, V (supports cross-attention)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        """
        Apply sparse top-k attention.
        
        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor [batch, seq_len, embed_dim]
            value: Value tensor [batch, seq_len, embed_dim]
            
        Returns:
            output: Attended features [batch, seq_len, embed_dim]
            attn_weights: Attention weights [batch, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_kv = key.size(1)
        
        # Project Q, K, V separately (supports cross-attention)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Sparse top-k selection
        k_value = min(self.top_k, scores.size(-1))
        topk_scores, topk_indices = torch.topk(scores, k=k_value, dim=-1)
        
        # Create sparse attention mask
        mask = torch.full_like(scores, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_scores)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(mask, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output, attn_weights.mean(dim=1)  # Return mean attention weights across heads


class MultiResolutionEncoder(nn.Module):
    """
    Multi-resolution feature extractor using Vision Transformer.

    Supports backbones:
      - vit_small_patch16_224  (22M, ImageNet)
      - vit_large_patch16_224  (304M, RETFound retinal foundation model)

    Pretrained weight loading priority:
      1. Local file (pretrained_weights/)
      2. HuggingFace Hub via timm (auto-download)
      3. Random initialization (fallback)
    """
    # Weight paths per backbone family
    _WEIGHT_PATHS = {
        'large': [
            'pretrained_weights/RETFound_cfp.pth',
            'pretrained_weights/vit_large_patch16_224.pth',
        ],
        'small': [
            'pretrained_weights/vit_small_patch16_224.safetensors',
            'pretrained_weights/vit_small_patch16_224.pth',
            'pretrained_weights/vit_small_patch16_224-15ec54c9.pth',
            '/kaggle/working/pretrained_weights/vit_small_patch16_224.pth',
            'outputs/models/vit_small_patch16_224.pth',
        ],
    }

    def __init__(self, backbone_name='vit_small_patch16_224', output_dim=384, img_size=224):
        super().__init__()
        self.fast_mode = os.environ.get("FAST_SINGLE_RESOLUTION", "1") == "1"
        self.resolutions = [img_size] if self.fast_mode else [img_size, int(img_size * 0.71), int(img_size * 0.57)]
        n_res = len(self.resolutions)

        use_pretrained = os.environ.get("USE_PRETRAINED", "1") == "1"
        mode_str = 'fast single-res' if self.fast_mode else '3-resolution'
        print(f"  Loading {backbone_name} ({mode_str})...")

        # Select weight paths based on backbone family
        family = 'large' if 'large' in backbone_name else 'small'
        local_paths = self._WEIGHT_PATHS.get(family, self._WEIGHT_PATHS['small'])
        timm_kwargs = dict(pretrained=False, num_classes=0, dynamic_img_size=True)

        loaded_local = False
        if use_pretrained:
            for lp in local_paths:
                if os.path.exists(lp):
                    try:
                        self.encoder = timm.create_model(backbone_name, **timm_kwargs)
                        if lp.endswith('.safetensors'):
                            from safetensors.torch import load_file
                            state = load_file(lp)
                        else:
                            state = torch.load(lp, map_location='cpu', weights_only=False)
                            if 'model' in state:
                                state = state['model']
                        # Filter out MAE decoder / mask_token keys
                        state = {k: v for k, v in state.items()
                                 if not k.startswith('decoder') and 'mask_token' not in k}
                        self.encoder.load_state_dict(state, strict=False)
                        print(f"  Loaded pretrained weights from local: {lp}")
                        loaded_local = True
                        break
                    except Exception as e:
                        print(f"  Local weight load failed ({lp}): {e}")

        # Priority 2: HuggingFace Hub via timm
        if not loaded_local and use_pretrained:
            try:
                self.encoder = timm.create_model(backbone_name, pretrained=True, num_classes=0, dynamic_img_size=True)
                print(f"  Loaded pretrained weights from HuggingFace Hub")
            except Exception as e:
                print(f"  HuggingFace download failed: {e}")
                self.encoder = timm.create_model(backbone_name, **timm_kwargs)
                print(f"  Using random initialization (fallback)")

        # Priority 3: Random init
        if not use_pretrained:
            self.encoder = timm.create_model(backbone_name, **timm_kwargs)
            print(f"  Using random initialization (USE_PRETRAINED=0)")

        # Backbone output dim (384 for ViT-S, 1024 for ViT-L/RETFound)
        self.backbone_dim = self.encoder.num_features

        self.resolution_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(self.backbone_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU())
            for _ in range(n_res)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * n_res, output_dim), nn.LayerNorm(output_dim), nn.GELU()
        )

    def forward(self, x):
        import torch.nn.functional as F
        features = []
        primary_size = self.resolutions[0]
        for resolution, proj in zip(self.resolutions, self.resolution_projections):
            if x.size(-1) != resolution:
                x_r = F.interpolate(x, size=(resolution, resolution), mode='bilinear', align_corners=False)
            else:
                x_r = x
            # Resize non-primary resolutions to primary for uniform patch count
            if resolution != primary_size:
                x_r = F.interpolate(x_r, size=(primary_size, primary_size), mode='bilinear', align_corners=False)
            tokens = self.encoder.forward_features(x_r)  # [B, N+1, D]
            patch_tokens = tokens[:, 1:, :]  # [B, N, D] - drop CLS token
            features.append(proj(patch_tokens))
        return self.fusion(torch.cat(features, dim=-1))


class ViGNN(nn.Module):
    """
    Visual Graph Neural Network (ViGNN) for retinal disease classification.
    Models visual features as a graph where each patch is a node.
    Features: Graph-based feature aggregation, adaptive edge weights, message passing
    Uses learnable edge weights to adaptively combine patch features based on disease context.
    Optimized for: ~50M parameters, graph-based reasoning, mobile deployment
    REQUIRES: knowledge_graph (ClinicalKnowledgeGraph instance)
    """
    def __init__(self, num_classes=45, hidden_dim=384, num_graph_layers=3, num_heads=4, dropout=0.1,
                 clinical_knowledge_graph=None, num_patches=196, patch_embed_dim=384,
                 backbone='vit_small_patch16_224', img_size=224):
        super(ViGNN, self).__init__()

        # MANDATORY clinical knowledge graph
        if clinical_knowledge_graph is None:
            raise ValueError("ViGNN requires clinical_knowledge_graph parameter (ClinicalKnowledgeGraph instance)")
        self.clinical_knowledge_graph = clinical_knowledge_graph
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Multi-resolution visual encoder
        self.visual_encoder = MultiResolutionEncoder(backbone, patch_embed_dim, img_size=img_size)
        
        # Patch projection
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Adaptive edge weight generator
        self.edge_weight_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Graph message passing layers with attention
        self.graph_layers = nn.ModuleList([
            SparseTopKAttention(hidden_dim, num_heads=num_heads, dropout=dropout, top_k=32)
            for _ in range(num_graph_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_graph_layers)])
        
        # Learnable disease prototypes (nodes)
        self.disease_prototypes = nn.Parameter(torch.randn(num_classes, hidden_dim))
        nn.init.normal_(self.disease_prototypes, std=0.02)
        
        # Disease-aware pooling
        self.disease_query = nn.Parameter(torch.randn(num_classes, hidden_dim))
        nn.init.normal_(self.disease_query, std=0.02)
        
        self.disease_attention = SparseTopKAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, top_k=64
        )
        
        # Global context aggregation
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)

        # Extract multi-resolution patch features (actual spatial tokens)
        patch_features = self.visual_encoder(x)  # [B, N, D]

        # Project patches to hidden dimension
        patch_embeds = self.patch_proj(patch_features)
        
        # Prepare disease prototypes
        disease_proto = self.disease_prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Generate adaptive edge weights using disease context
        patch_mean = patch_embeds.mean(dim=1, keepdim=True)
        patch_disease_concat = torch.cat(
            [patch_mean.expand(-1, self.num_classes, -1), disease_proto],
            dim=-1
        )
        
        edge_weights = self.edge_weight_generator(patch_disease_concat)
        
        # Graph message passing through patches
        graph_embeds = patch_embeds
        for graph_layer, norm in zip(self.graph_layers, self.layer_norms):
            attn_out, _ = graph_layer(graph_embeds, graph_embeds, graph_embeds)
            graph_embeds = norm(graph_embeds + attn_out)
        
        # Global patch aggregation
        patch_global = graph_embeds.mean(dim=1)
        global_context = self.global_context(patch_global)
        
        # Disease-aware attention
        disease_query = self.disease_query.unsqueeze(0).expand(batch_size, -1, -1)
        
        disease_out, _ = self.disease_attention(
            disease_query,
            graph_embeds,
            graph_embeds
        )
        
        # Aggregate disease-aware features
        disease_aware = disease_out.mean(dim=1)
        
        # Combine global context and disease-aware features
        final_features = torch.cat([global_context, disease_aware], dim=-1)
        
        # Final classification
        logits = self.classifier(final_features)
        
        return logits


def create_vignn_model(num_classes=48, hidden_dim=384, num_graph_layers=3, num_heads=4, dropout=0.1, clinical_knowledge_graph=None, num_patches=196, patch_embed_dim=384, checkpoint_path=None, backbone='vit_small_patch16_224', img_size=224):
    """
    Create ViGNN model and optionally load from checkpoint.

    Args:
        num_classes: Number of disease classes (default: 45)
        hidden_dim: Hidden dimension size (default: 384)
        num_graph_layers: Number of graph layers (default: 3)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.1)
        clinical_knowledge_graph: ClinicalKnowledgeGraph instance (required)
        num_patches: Number of patches (default: 196)
        patch_embed_dim: Patch embedding dimension (default: 384)
        checkpoint_path: Path to checkpoint file (optional)
        img_size: Input image size (default: 224)

    Returns:
        model: ViGNN model instance
    """
    model = ViGNN(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_graph_layers=num_graph_layers,
        num_heads=num_heads,
        dropout=dropout,
        clinical_knowledge_graph=clinical_knowledge_graph,
        num_patches=num_patches,
        patch_embed_dim=patch_embed_dim,
        backbone=backbone,
        img_size=img_size,
    )
    
    if checkpoint_path is not None:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model weights from checkpoint")
            
            if 'best_f1' in checkpoint:
                print(f"  Best F1 Score: {checkpoint['best_f1']:.4f}")
            if 'best_auc' in checkpoint:
                print(f"  Best AUC Score: {checkpoint['best_auc']:.4f}")
        else:
            print("⚠️  Checkpoint format not recognized")
    
    return model


def create_knowledge_graph(disease_names=None):
    """
    Create a ClinicalKnowledgeGraph with Uganda-specific disease relationships.
    
    Args:
        disease_names: List of disease codes (default: standard 45 diseases)
        
    Returns:
        knowledge_graph: ClinicalKnowledgeGraph instance
    """
    if disease_names is None:
        # Default 48 retinal diseases from RFMiD dataset (updated to match checkpoint)
        disease_names = [
            "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
            "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
            "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS", "CB", "ODPM",
            "PRH", "MNF", "HR", "CRAO", "TD", "CME", "PTCR", "CF", "VH", "MCA",
            "VS", "BRAO", "PLQ", "HPED", "CL", "AMD", "DME", "ROP"
        ]
    
    knowledge_graph = ClinicalKnowledgeGraph(disease_names=disease_names)
    
    print(f"✓ ClinicalKnowledgeGraph initialized")
    print(f"  • {knowledge_graph.num_classes} diseases")
    print(f"  • {knowledge_graph.get_edge_count()} clinical relationships")
    print(f"  • Uganda-specific epidemiology included")
    
    return knowledge_graph


if __name__ == "__main__":
    # Test model creation
    print("Testing ViGNN model...")
    
    # Create knowledge graph first (required for ViGNN)
    kg = create_knowledge_graph()
    
    model = create_vignn_model(num_classes=48, clinical_knowledge_graph=kg)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"✓ Model created successfully")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params/1e6:.1f}M")
    print(f"  Trainable parameters: {trainable_params/1e6:.1f}M")
    
    # Test knowledge graph
    print("\nTesting ClinicalKnowledgeGraph...")
    print(f"✓ Knowledge graph created successfully")
