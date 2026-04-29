# RetinalAI - Video Demo Script

## Video Details
- **Title**: RetinalAI: AI-Powered Retinal Disease Screening with Clinical Knowledge Graph Reasoning
- **Duration**: 8-10 minutes
- **Audience**: Hackathon judges, investors, clinical stakeholders, YouTube viewers
- **Tone**: Professional, confident, concise. Show don't tell.

---

## SCENE 1: Hook (0:00 - 0:30)

**[Screen: Black with text fading in]**

> "285 million people worldwide suffer from visual impairment.
> 80% of blindness is preventable — if detected early.
> In Sub-Saharan Africa, there is 1 ophthalmologist per 1 million people."

**[Cut to: RetinalAI dashboard loading]**

**NARRATION:**
"What if a single retinal photograph could screen for 45 eye diseases in under 200 milliseconds — with clinical reasoning that explains every prediction? This is RetinalAI."

---

## SCENE 2: The Problem (0:30 - 1:30)

**[Screen: Show statistics or simple slides]**

**NARRATION:**
"Retinal diseases like diabetic retinopathy, glaucoma, and macular degeneration are the leading causes of preventable blindness. In Uganda and across Sub-Saharan Africa, patients wait months for specialist screening. By the time they see a doctor, irreversible damage has already occurred.

The challenge is not just detection — it's making AI that clinicians can trust. That means explainability, clinical reasoning, and production-grade reliability.

We built RetinalAI to solve this."

---

## SCENE 3: Live Demo — Upload & Predict (1:30 - 3:30)

**[Screen: Frontend at localhost:3330, Screening page]**

**ACTION: Upload a retinal fundus image**

**NARRATION:**
"Let me show you the system in action. I'll upload a retinal fundus photograph from the RFMiD clinical dataset."

**[Drag and drop image → Click 'Analyze Image']**

"The image first passes through our three-layer fundus validation gate — checking structure, statistical properties, and confirming this is actually a retinal photograph, not a selfie or a cat picture."

**[Results appear: diseases detected, referral priority, inference time]**

"In 205 milliseconds, the model detected 11 potential conditions and assigned an URGENT referral priority. The system identified Macular Scars, Retinitis, and Optic Disc Pallor with medium confidence."

**[Point to the confidence bars and referral badge]**

"Every prediction shows the exact confidence score. The referral priority — URGENT, ROUTINE, or FOLLOW-UP — is determined by our Clinical Knowledge Graph, not just raw probabilities."

---

## SCENE 4: Clinical Reasoning (3:30 - 4:30)

**[Screen: Scroll to Clinical Reasoning section]**

**NARRATION:**
"This is where RetinalAI goes beyond simple classification. Our Clinical Knowledge Graph encodes 144 disease relationships from peer-reviewed ophthalmology literature, specifically calibrated for Uganda's disease epidemiology."

**[Point to KG adjustments showing before/after probabilities]**

"When the model detects Diabetic Retinopathy with high confidence, the knowledge graph automatically boosts the probability of related conditions — Cystoid Macular Edema, Vitreous Hemorrhage — because these diseases co-occur clinically. This is clinical reasoning, not just pattern matching."

**[Show Visual Findings and Treatment Recommendations]**

"The system also shows expected visual findings — microaneurysms, hemorrhages, hard exudates — so the clinician knows exactly what to look for. And it provides evidence-based treatment considerations for each detected condition."

---

## SCENE 5: Explainability — 5 Methods (4:30 - 5:30)

**[Screen: Navigate to Screening page, expand explainability panel]**

**NARRATION:**
"Trust in medical AI requires explainability. RetinalAI offers five complementary explanation methods."

**[Click GradCAM tab]**
"GradCAM shows exactly which retinal regions drove each prediction — here you can see the model focused on the macular area for this diagnosis."

**[Click LIME tab]**
"LIME generates superpixel explanations — these green regions are the most influential areas for the prediction."

**[Mention other tabs]**
"We also support SHAP values, Integrated Gradients, and ELI5 — giving clinicians multiple lenses to validate the AI's reasoning."

---

## SCENE 6: Disease Knowledge Base (5:30 - 6:15)

**[Screen: Click on a detected disease to expand the detail card]**

**NARRATION:**
"Each disease has a comprehensive clinical information card. For Diabetic Retinopathy — severity level 3, VASCULAR category. Risk factors: diabetes, hypertension, high cholesterol. Treatment options: glycemic control, laser photocoagulation, anti-VEGF injections. Urgency: immediate referral within 24 to 48 hours.

This turns the AI from a black box into a clinical decision support tool."

**[Scroll to Knowledge Graph panel]**

"The Knowledge Graph explorer shows all 9 disease categories, Uganda-specific prevalence data, severity levels, and co-occurrence relationships — giving clinicians the complete epidemiological context."

---

## SCENE 7: Architecture & Models (6:15 - 7:00)

**[Screen: Navigate to System page or show architecture diagram]**

**NARRATION:**
"Under the hood, RetinalAI evaluated four custom graph neural network architectures in a rigorous 3-fold cross-validation:

- ViGNN — Visual Graph Neural Network with disease prototypes
- GraphCLIP — dynamic graph learning with sparse attention
- VisualLanguageGNN — cross-modal visual-text fusion
- SceneGraphTransformer — our winner — with 3 ensemble branches and uncertainty calibration

All four models share a pretrained Vision Transformer backbone, a Clinical Knowledge Graph with 144 relationships, and sparse top-k attention for mobile efficiency.

The SceneGraphTransformer won with the highest F1 score and 80% sample accuracy. It's exported to both ONNX and TorchScript for cross-platform deployment."

---

## SCENE 8: MLOps Pipeline (7:00 - 7:45)

**[Screen: Show terminal or dashboard metrics]**

**NARRATION:**
"This isn't just a model — it's a production MLOps pipeline.

Data ingestion with automated validation — 7 quality checks on every dataset.
Training with differential learning rates, EMA, mixed precision, and early stopping.
Model comparison with weighted scoring — F1 at 40%, AUC at 30%, Precision and Recall at 15% each.
Export to ONNX and TorchScript with quantization benchmarks.
Monitoring with drift detection, SLA compliance tracking, and prediction logging.
Governance with model cards, fairness evaluation, and a human-in-the-loop review queue.

Every prediction is logged. Every model version is tracked. Every decision is auditable."

**[Show the live performance panel]**

"Right now, the system is running with sub-200ms inference, zero error rate, and full SLA compliance."

---

## SCENE 9: Tech Stack (7:45 - 8:15)

**[Screen: Show README or architecture slide]**

**NARRATION:**
"The tech stack:

Backend — FastAPI with UV, JWT authentication, structured logging, rate limiting.
Frontend — Next.js 16 with Bun, Zustand for state, TanStack Query for data, fully responsive PWA.
Training — PyTorch DDP across 8 NVIDIA RTX A6000 GPUs.
Infrastructure — Docker Compose, DVC for data versioning, GitHub Actions CI/CD.
68 automated tests. IEEE publication-quality plots. EU AI Act compliance readiness.

Built for production. Built for scale. Built for impact."

---

## SCENE 10: Impact & Close (8:15 - 9:00)

**[Screen: Dashboard with RetinalAI logo]**

**NARRATION:**
"RetinalAI is designed for the places that need it most — community health centers in Uganda, mobile screening camps, primary care clinics with no ophthalmologist.

One photograph. 45 diseases. 200 milliseconds. Clinical reasoning that a doctor can trust.

Early detection saves sight. RetinalAI makes it possible at scale."

**[Screen: Contact info / GitHub repo / QR code]**

"Try it yourself. The code is open source. Thank you."

---

## RECORDING TIPS

### Before Recording
- [ ] Backend running on port 8088 with model loaded
- [ ] Frontend running on port 3330
- [ ] Have 3-4 retinal fundus images ready (mix of healthy and diseased)
- [ ] Clear browser cache for clean demo
- [ ] Close unnecessary tabs and notifications
- [ ] Set screen resolution to 1920x1080

### During Recording
- [ ] Use a clean, steady mouse — avoid fast scrolling
- [ ] Pause 1-2 seconds after each action to let viewers read
- [ ] Keep narration slightly ahead of on-screen action
- [ ] Zoom in (Ctrl +) when showing small text or metrics
- [ ] Use picture-in-picture for face camera (optional)

### Screen Flow
1. Dashboard (overview metrics) → 2. Screening (upload + predict) → 3. Scroll to clinical reasoning → 4. Show explainability tabs → 5. Expand disease cards → 6. Knowledge graph panel → 7. System page (architecture) → 8. Terminal (show training logs briefly) → 9. Back to dashboard (close)

### Audio
- Record narration separately for clean audio (recommended)
- Or use a good USB microphone in a quiet room
- Background music: subtle, royalty-free, medical/tech genre
- Keep volume at 15-20% under narration

### Editing
- Add lower-third text labels for each section
- Add subtle zoom transitions between scenes
- Include brief text overlays for key numbers (45 diseases, 200ms, 144 relationships)
- End card with GitHub link and contact info
- Total cut duration: aim for 8 minutes (trim pauses)

---

## HACKATHON PITCH VERSION (3 minutes)

For a 3-minute hackathon pitch, use only:
- Scene 1 (Hook) — 20 seconds
- Scene 2 (Problem) — 30 seconds
- Scene 3 (Live Demo) — 60 seconds (upload, predict, show results)
- Scene 5 (Explainability) — 30 seconds (GradCAM only)
- Scene 7 (Architecture) — 20 seconds (one sentence per model)
- Scene 10 (Impact) — 20 seconds

Skip: Clinical reasoning deep-dive, MLOps pipeline details, tech stack.
Focus on: the problem, the demo, and the impact.

---

## INVESTOR PITCH VERSION (5 minutes)

Add to the hackathon version:
- Scene 4 (Clinical Reasoning) — 45 seconds (knowledge graph is the differentiator)
- Scene 8 (MLOps) — 30 seconds (production-ready, not a prototype)
- Market size: $3.2B ophthalmic AI market, growing 30% CAGR
- Business model: SaaS per-scan pricing for clinics, partnership with MOH Uganda
- Ask: seed funding for clinical validation study and regulatory pathway
