"""Clinical reasoning, disease info, and knowledge graph endpoints."""
from fastapi import APIRouter, HTTPException
from backend.app.core.model_service import model_service, DISEASE_NAMES

from fastapi import Depends
from backend.app.core.feature_gate import require_tier

router = APIRouter(
    prefix="/api/v1/clinical",
    tags=["clinical"],
    dependencies=[Depends(require_tier("clinician", feature="clinical_reasoning"))],
)


DISEASE_INFO = {
    "DR": {"severity": 3, "category": "VASCULAR", "description": "Damage to retinal blood vessels caused by diabetes. Leading cause of blindness in working-age adults.", "risk_factors": ["Diabetes", "Hypertension", "High cholesterol", "Pregnancy"], "treatment": ["Glycemic control", "Laser photocoagulation", "Anti-VEGF injections", "Vitrectomy for advanced cases"], "urgency": "Immediate referral within 24-48h"},
    "ARMD": {"severity": 2, "category": "DEGENERATIVE", "description": "Progressive deterioration of the macula causing central vision loss. Most common cause of vision loss in adults over 50.", "risk_factors": ["Age >50", "Smoking", "Family history", "UV exposure", "Obesity"], "treatment": ["AREDS supplements", "Anti-VEGF for wet AMD", "Low vision aids", "Photodynamic therapy"], "urgency": "Routine referral within 2 weeks"},
    "MH": {"severity": 2, "category": "STRUCTURAL", "description": "Full-thickness defect in the foveal center causing central vision distortion and decreased acuity.", "risk_factors": ["Age >60", "Female sex", "Myopia", "Trauma", "Previous vitrectomy"], "treatment": ["Vitrectomy with ILM peeling", "Ocriplasmin injection", "Observation for small holes"], "urgency": "Routine referral within 2 weeks"},
    "DN": {"severity": 2, "category": "VASCULAR", "description": "Diabetic neuropathy affecting the retinal nerve fiber layer, often co-occurring with diabetic retinopathy.", "risk_factors": ["Diabetes duration >10 years", "Poor glycemic control", "Hypertension"], "treatment": ["Strict glycemic control", "Neuroprotective agents", "Regular monitoring"], "urgency": "Routine referral"},
    "MYA": {"severity": 1, "category": "DEGENERATIVE", "description": "Retinal changes associated with high myopia including lattice degeneration and posterior staphyloma.", "risk_factors": ["High myopia (>-6D)", "Axial length >26mm", "Family history"], "treatment": ["Regular retinal screening", "Atropine for progression", "Scleral reinforcement in severe cases"], "urgency": "Annual screening recommended"},
    "BRVO": {"severity": 2, "category": "VASCULAR", "description": "Blockage of a branch retinal vein causing sectoral hemorrhages and vision changes.", "risk_factors": ["Hypertension", "Diabetes", "Hyperlipidemia", "Atherosclerosis"], "treatment": ["Anti-VEGF if macular edema", "Sectoral laser photocoagulation", "Monitor for neovascularization"], "urgency": "Referral within 1 week"},
    "TSLN": {"severity": 1, "category": "STRUCTURAL", "description": "Tessellation of the fundus where choroidal vessels become visible through a thin retinal pigment epithelium.", "risk_factors": ["Myopia", "Aging", "Thin choroid"], "treatment": ["No treatment required", "Monitor for pathologic myopia changes"], "urgency": "Observation only"},
    "ERM": {"severity": 1, "category": "STRUCTURAL", "description": "Fibrocellular membrane on the inner retinal surface causing metamorphopsia and reduced vision.", "risk_factors": ["Age >50", "Previous retinal surgery", "Retinal vascular disease", "Inflammation"], "treatment": ["Observation if mild", "Vitrectomy with membrane peel for symptomatic cases"], "urgency": "Routine referral if symptomatic"},
    "LS": {"severity": 1, "category": "STRUCTURAL", "description": "Chorioretinal scars from previous laser photocoagulation treatment.", "risk_factors": ["Previous DR treatment", "Previous retinal tear treatment"], "treatment": ["No active treatment needed", "Monitor underlying condition"], "urgency": "Routine follow-up"},
    "MS": {"severity": 1, "category": "STRUCTURAL", "description": "Scarring of the macular region from previous disease or injury affecting central vision.", "risk_factors": ["Previous CNV", "Trauma", "Infection", "Age-related macular degeneration"], "treatment": ["Low vision rehabilitation", "Monitor for recurrent activity"], "urgency": "Routine follow-up"},
    "CSR": {"severity": 1, "category": "DEGENERATIVE", "description": "Serous detachment of the neurosensory retina at the macula, typically self-limiting.", "risk_factors": ["Male sex", "Age 20-50", "Stress", "Type A personality", "Corticosteroid use"], "treatment": ["Observation (80% resolve spontaneously)", "Laser if chronic", "PDT for chronic cases"], "urgency": "Routine referral"},
    "ODC": {"severity": 2, "category": "GLAUCOMATOUS", "description": "Enlarged optic cup suggesting glaucomatous damage to optic nerve fibers.", "risk_factors": ["Elevated IOP", "Family history", "Age >40", "Myopia", "African ancestry"], "treatment": ["IOP-lowering drops", "Laser trabeculoplasty", "Filtration surgery if progressive"], "urgency": "Referral within 2 weeks"},
    "CRVO": {"severity": 3, "category": "VASCULAR", "description": "Blockage of the central retinal vein causing sudden painless vision loss with widespread hemorrhages.", "risk_factors": ["Hypertension", "Diabetes", "Glaucoma", "Hypercoagulable states", "Age >50"], "treatment": ["Anti-VEGF injections", "Intravitreal steroids", "Pan-retinal photocoagulation for ischemic type", "Treat underlying cause"], "urgency": "Urgent referral within 24h"},
    "TV": {"severity": 1, "category": "VASCULAR", "description": "Abnormally tortuous retinal blood vessels which may indicate systemic vascular disease.", "risk_factors": ["Hypertension", "Diabetes", "Congenital conditions"], "treatment": ["Investigate and treat underlying systemic condition", "Regular monitoring"], "urgency": "Routine evaluation"},
    "AH": {"severity": 1, "category": "STRUCTURAL", "description": "Calcium-lipid bodies floating in the vitreous cavity, usually benign and incidental.", "risk_factors": ["Age", "Diabetes", "Hypercholesterolemia"], "treatment": ["No treatment usually required", "Vitrectomy only if severely impairs vision"], "urgency": "Observation only"},
    "ODP": {"severity": 2, "category": "GLAUCOMATOUS", "description": "Pale optic disc indicating optic nerve damage from glaucoma, ischemia, or other causes.", "risk_factors": ["Glaucoma", "Optic neuritis", "Ischemic optic neuropathy", "Compressive lesion"], "treatment": ["Investigate cause", "IOP management if glaucomatous", "Neuroimaging if non-glaucomatous"], "urgency": "Urgent evaluation"},
    "ODE": {"severity": 2, "category": "GLAUCOMATOUS", "description": "Swelling of the optic disc which may indicate raised intracranial pressure or optic neuritis.", "risk_factors": ["Raised ICP", "Optic neuritis", "Malignant hypertension", "Central vein occlusion"], "treatment": ["Urgent neuroimaging", "Treat underlying cause", "ICP management if papilledema"], "urgency": "Urgent - rule out raised ICP"},
    "ST": {"severity": 1, "category": "VASCULAR", "description": "Optociliary shunt vessels on the optic disc, may indicate chronic venous obstruction.", "risk_factors": ["Optic nerve sheath meningioma", "Chronic papilledema", "Central retinal vein occlusion"], "treatment": ["Neuroimaging to rule out compressive lesion", "Monitor"], "urgency": "Routine evaluation with imaging"},
    "AION": {"severity": 3, "category": "VASCULAR", "description": "Acute ischemic event of the optic nerve head causing sudden vision loss. Arteritic form is an emergency.", "risk_factors": ["Giant cell arteritis", "Hypertension", "Diabetes", "Sleep apnea", "Age >50"], "treatment": ["Immediate ESR/CRP if arteritic suspected", "High-dose IV steroids for GCA", "Aspirin for non-arteritic"], "urgency": "EMERGENCY if arteritic - immediate steroids"},
    "PT": {"severity": 1, "category": "VASCULAR", "description": "Ectatic parafoveal capillaries causing macular edema and vision loss.", "risk_factors": ["Age >40", "Bilateral presentation common"], "treatment": ["Observation", "Anti-VEGF for macular edema", "Focal laser"], "urgency": "Routine referral"},
    "RT": {"severity": 2, "category": "STRUCTURAL", "description": "Tractional forces on the retina from fibrovascular membranes, common in advanced diabetic eye disease.", "risk_factors": ["Proliferative diabetic retinopathy", "Previous vitreous hemorrhage", "Trauma"], "treatment": ["Vitrectomy if involving or threatening macula", "Anti-VEGF as adjunct", "Observation if stable and away from macula"], "urgency": "Urgent if macula-involving or progressive"},
    "RS": {"severity": 2, "category": "INFECTIOUS_IMMUNOLOGIC", "description": "Inflammation of the retina, often infectious. CMV retinitis in immunocompromised patients.", "risk_factors": ["HIV/AIDS", "Immunosuppression", "CMV infection", "Toxoplasmosis"], "treatment": ["Antiviral therapy for CMV", "Anti-toxoplasma therapy", "Intravitreal antivirals"], "urgency": "Urgent referral - sight-threatening"},
    "CRS": {"severity": 1, "category": "STRUCTURAL", "description": "Chorioretinal scars from previous inflammatory or infectious episodes.", "risk_factors": ["Previous toxoplasmosis", "Previous CMV", "Histoplasmosis"], "treatment": ["Monitor for reactivation", "No active treatment for inactive scars"], "urgency": "Routine monitoring"},
    "EDN": {"severity": 2, "category": "STRUCTURAL", "description": "Exudative retinal detachment from fluid accumulation under the neurosensory retina.", "risk_factors": ["Vogt-Koyanagi-Harada disease", "Posterior scleritis", "Choroidal tumors", "Severe preeclampsia"], "treatment": ["Treat underlying cause", "Systemic steroids for VKH", "Urgent if tumor suspected"], "urgency": "Urgent evaluation"},
    "RPEC": {"severity": 1, "category": "DEGENERATIVE", "description": "Retinal pigment epithelium changes that may indicate early macular degeneration or other conditions.", "risk_factors": ["Age", "UV exposure", "Smoking", "Genetic predisposition"], "treatment": ["Monitor for progression", "AREDS supplements if intermediate AMD", "Lifestyle modifications"], "urgency": "Routine screening"},
    "MHL": {"severity": 1, "category": "STRUCTURAL", "description": "Partial-thickness defect of the fovea, less severe than full-thickness macular hole.", "risk_factors": ["Age", "Epiretinal membrane", "Vitreomacular traction"], "treatment": ["Observation in most cases", "Vitrectomy if progressing to full-thickness hole"], "urgency": "Routine monitoring"},
    "RP": {"severity": 2, "category": "DEGENERATIVE", "description": "Group of inherited retinal dystrophies causing progressive photoreceptor degeneration and night blindness.", "risk_factors": ["Family history", "Genetic mutations (RHO, USH2A)", "Consanguinity"], "treatment": ["Vitamin A supplementation", "Low vision aids", "Gene therapy (Luxturna for RPE65)", "Retinal prosthesis research"], "urgency": "Genetic counseling referral"},
    "CWS": {"severity": 2, "category": "INFECTIOUS_IMMUNOLOGIC", "description": "White fluffy lesions representing nerve fiber layer infarcts. Can indicate HIV retinopathy in endemic areas.", "risk_factors": ["HIV/AIDS (CD4 <200)", "Hypertension", "Diabetes", "Lupus"], "treatment": ["Optimize HAART", "Treat underlying condition", "Monitor for progression"], "urgency": "Investigate systemic cause"},
    "CB": {"severity": 2, "category": "VASCULAR", "description": "Coats disease - idiopathic retinal telangiectasia with subretinal exudation, typically unilateral.", "risk_factors": ["Male sex", "Young age", "Unilateral presentation"], "treatment": ["Laser photocoagulation", "Cryotherapy", "Anti-VEGF for exudation", "Surgery for advanced disease"], "urgency": "Referral within 1-2 weeks"},
    "ODPM": {"severity": 2, "category": "GLAUCOMATOUS", "description": "Maculopathy associated with congenital optic disc pit causing serous macular detachment.", "risk_factors": ["Congenital optic disc pit", "Young adults"], "treatment": ["Observation", "Laser barrage", "Vitrectomy with gas tamponade"], "urgency": "Routine referral"},
    "PRH": {"severity": 2, "category": "VASCULAR", "description": "Blood between the internal limiting membrane and posterior hyaloid, often from proliferative DR.", "risk_factors": ["Proliferative diabetic retinopathy", "Valsalva retinopathy", "Blood dyscrasias"], "treatment": ["Observation if small", "Nd:YAG hyaloidotomy", "Vitrectomy if persistent"], "urgency": "Evaluate underlying cause"},
    "MNF": {"severity": 1, "category": "STRUCTURAL", "description": "Congenital anomaly of myelin extending beyond the lamina cribrosa onto retinal nerve fibers.", "risk_factors": ["Congenital", "Usually incidental finding"], "treatment": ["No treatment required", "Document for future reference"], "urgency": "Observation only"},
    "HR": {"severity": 2, "category": "VASCULAR", "description": "Retinal vascular changes caused by systemic hypertension including AV nicking and flame hemorrhages.", "risk_factors": ["Chronic hypertension", "Malignant hypertension", "Renal disease"], "treatment": ["Blood pressure control", "Treat underlying hypertension", "Monitor for complications"], "urgency": "Urgent if malignant hypertension"},
    "CRAO": {"severity": 3, "category": "VASCULAR", "description": "Acute blockage of the central retinal artery causing sudden painless vision loss. Ophthalmic emergency.", "risk_factors": ["Cardiovascular disease", "Giant cell arteritis", "Carotid stenosis", "Atrial fibrillation"], "treatment": ["EMERGENCY: ocular massage, anterior chamber paracentesis", "tPA within 4.5h", "Full stroke workup", "Cardiovascular evaluation"], "urgency": "EMERGENCY - immediate referral within minutes"},
    "TD": {"severity": 1, "category": "STRUCTURAL", "description": "Congenital anomaly of the optic disc appearing tilted, may cause visual field defects mimicking glaucoma.", "risk_factors": ["Congenital", "Myopia", "Astigmatism"], "treatment": ["No treatment needed", "Differentiate from glaucoma", "Regular visual field monitoring"], "urgency": "Observation only"},
    "CME": {"severity": 2, "category": "VASCULAR", "description": "Fluid accumulation in the macula in a petaloid pattern causing blurred central vision.", "risk_factors": ["Diabetes", "Post-cataract surgery (Irvine-Gass)", "Uveitis", "Vein occlusion", "Prostaglandin use"], "treatment": ["Anti-VEGF injections", "Intravitreal steroids", "Topical NSAIDs", "Treat underlying cause"], "urgency": "Referral within 1 week"},
    "PTCR": {"severity": 1, "category": "STRUCTURAL", "description": "Chorioretinal changes following ocular or head trauma.", "risk_factors": ["Blunt ocular trauma", "Whiplash injury", "Head trauma"], "treatment": ["Monitor for complications", "Treat specific sequelae (retinal detachment, vitreous hemorrhage)"], "urgency": "Routine follow-up unless acute"},
    "CF": {"severity": 1, "category": "STRUCTURAL", "description": "Folding of the choroid and overlying retina, may indicate orbital mass or hypotony.", "risk_factors": ["Orbital mass", "Hypotony", "Posterior scleritis", "Papilledema"], "treatment": ["Investigate underlying cause", "Orbital imaging if mass suspected"], "urgency": "Evaluate for orbital pathology"},
    "VH": {"severity": 2, "category": "VASCULAR", "description": "Blood in the vitreous cavity obscuring vision. Often secondary to proliferative diabetic retinopathy.", "risk_factors": ["Proliferative DR", "Retinal tears", "Trauma", "PVD with retinal break"], "treatment": ["B-scan to rule out detachment", "Observation if mild", "Vitrectomy if non-clearing or retinal detachment", "Anti-VEGF"], "urgency": "Urgent - rule out retinal detachment"},
    "MCA": {"severity": 2, "category": "VASCULAR", "description": "Focal dilation of retinal arteriole, risk of hemorrhage. Associated with systemic hypertension.", "risk_factors": ["Hypertension", "Atherosclerosis", "Female sex", "Age >60"], "treatment": ["Focal laser if leaking", "Blood pressure control", "Monitor for rupture"], "urgency": "Routine referral"},
    "VS": {"severity": 2, "category": "INFECTIOUS_IMMUNOLOGIC", "description": "Inflammation of retinal blood vessels, may be primary or secondary to systemic disease.", "risk_factors": ["Sarcoidosis", "Behcet's disease", "Tuberculosis", "Multiple sclerosis", "SLE"], "treatment": ["Systemic workup", "Corticosteroids", "Immunosuppression for recurrent cases", "Treat underlying cause"], "urgency": "Urgent referral"},
    "BRAO": {"severity": 3, "category": "VASCULAR", "description": "Acute occlusion of a branch retinal artery causing sectoral vision loss.", "risk_factors": ["Carotid stenosis", "Cardiac emboli", "Giant cell arteritis", "Hypercoagulable states"], "treatment": ["Stroke workup", "Carotid evaluation", "Antiplatelet therapy", "ESR/CRP to rule out GCA"], "urgency": "Urgent - same-day evaluation"},
    "PLQ": {"severity": 1, "category": "STRUCTURAL", "description": "Calcified deposits (drusen) within the optic disc substance, usually congenital.", "risk_factors": ["Congenital", "Autosomal dominant inheritance"], "treatment": ["No treatment needed", "Monitor visual fields", "Differentiate from papilledema"], "urgency": "Observation only"},
    "HPED": {"severity": 2, "category": "VASCULAR", "description": "Hemorrhagic pigment epithelial detachment, often associated with neovascular AMD.", "risk_factors": ["Neovascular AMD", "PCV (polypoidal choroidal vasculopathy)", "Anticoagulant use"], "treatment": ["Anti-VEGF injections", "Surgical drainage for large hemorrhages", "Treat underlying CNV"], "urgency": "Urgent referral"},
    "CL": {"severity": 2, "category": "STRUCTURAL", "description": "Lesion within the choroid that may be benign (nevus) or malignant (melanoma). Requires careful evaluation.", "risk_factors": ["Fair skin", "UV exposure", "BAP1 mutations", "Oculodermal melanocytosis"], "treatment": ["Observation with serial imaging for small nevi", "Plaque brachytherapy or proton beam for melanoma", "Enucleation for large tumors"], "urgency": "Urgent evaluation - rule out malignancy"},
}


@router.get("/disease-info/{code}")
async def get_disease_info(code: str):
    """Get clinical information for a specific disease."""
    info = DISEASE_INFO.get(code)
    if not info:
        return {"code": code, "name": DISEASE_NAMES.get(code, code), "info_available": False}
    return {"code": code, "name": DISEASE_NAMES.get(code, code), "info_available": True, **info}


@router.get("/disease-info")
async def list_all_disease_info():
    """Get clinical info for all diseases with full metadata."""
    diseases = []
    for code in model_service.disease_codes:
        info = DISEASE_INFO.get(code)
        entry = {
            "code": code,
            "name": DISEASE_NAMES.get(code, code),
            "info_available": info is not None,
        }
        if info:
            entry.update(info)
        diseases.append(entry)
    return {"total": len(diseases), "diseases": diseases}


@router.get("/knowledge-graph")
async def get_knowledge_graph():
    """Get clinical knowledge graph data for visualization."""
    kg = model_service.kg
    if kg is None:
        return {"error": "Knowledge graph not loaded"}

    categories = {cat: list(diseases) for cat, diseases in kg.categories.items()}

    edges = []
    for disease, related in kg.cooccurrence.items():
        for r in related:
            edges.append({"source": disease, "target": r, "type": "co-occurrence"})

    severity = {d: kg.get_disease_severity(d) for d in kg.disease_names if kg.get_disease_severity(d) > 0}
    prevalence = dict(kg.uganda_prevalence)

    return {
        "diseases": len(kg.disease_names),
        "edges": len(edges),
        "categories": categories,
        "relationships": edges,
        "severity": severity,
        "prevalence": prevalence,
        "disease_names": {code: DISEASE_NAMES.get(code, code) for code in kg.disease_names},
    }


@router.post("/explain-reasoning")
async def explain_reasoning(predictions: dict[str, float]):
    """Explain clinical reasoning applied to predictions."""
    if not predictions:
        raise HTTPException(400, "Predictions cannot be empty")
    if len(predictions) > 50:
        raise HTTPException(400, "Too many predictions (max 50)")
    for key, val in predictions.items():
        if not isinstance(val, (int, float)):
            raise HTTPException(400, f"Invalid probability for {key}: must be a number")
        if not (0.0 <= val <= 1.0):
            raise HTTPException(400, f"Probability for {key} must be between 0 and 1")

    kg = model_service.kg
    if kg is None:
        return {"error": "Knowledge graph not loaded"}

    original = {k: v for k, v in predictions.items() if k in kg.disease_names}
    refined = kg.apply_clinical_reasoning(original)

    adjustments = []
    for disease in refined:
        if disease in original:
            diff = refined[disease] - original[disease]
            if abs(diff) > 0.001:
                adjustments.append({
                    "disease": disease,
                    "name": DISEASE_NAMES.get(disease, disease),
                    "original": round(original[disease], 4),
                    "refined": round(refined[disease], 4),
                    "boost": round(diff, 4),
                    "reason": _get_reasoning(disease, original),
                })

    detected = [d for d, p in refined.items() if p > 0.3]
    visual_findings = kg.get_visual_findings(detected)
    treatment = kg.get_treatment_recommendations(detected)
    referral = kg.get_referral_priority(detected)

    return {
        "adjustments": adjustments,
        "referral_priority": referral,
        "visual_findings": visual_findings,
        "treatment_recommendations": treatment,
        "detected_count": len(detected),
    }


def _get_reasoning(disease, predictions):
    """Generate human-readable reasoning for a prediction adjustment."""
    reasons = {
        "CME": "DR detected with high confidence - CME frequently co-occurs with diabetic retinopathy",
        "VH": "Vascular disease detected - vitreous hemorrhage risk elevated",
        "PRH": "DR/vascular pathology detected - preretinal hemorrhage associated",
        "CRVO": "Hypertensive retinopathy detected - vein occlusion risk increased",
        "BRVO": "Hypertensive retinopathy detected - branch vein occlusion risk increased",
        "CRAO": "Vascular disease detected - arterial occlusion risk elevated",
        "BRAO": "Vascular disease detected - branch arterial occlusion risk elevated",
        "MCA": "Hypertensive retinopathy detected - macroaneurysm associated",
        "ODE": "Optic disc cupping detected - disc edema may co-exist",
        "ODPM": "Optic disc cupping detected - pit maculopathy associated",
        "RPEC": "Cotton wool spots detected - RPE changes frequently co-occur",
        "HPED": "Inflammatory markers detected - hemorrhagic PED associated",
        "MNF": "Preretinal hemorrhage detected - myelinated nerve fibers associated",
    }
    return reasons.get(disease, "Clinical co-occurrence pattern detected by knowledge graph")
