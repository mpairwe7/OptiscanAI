# Commercialization Strategy: Retinal Disease AI Platform

**Classification**: Internal Strategy Document
**Version**: 1.0 | April 2026
**System**: Multi-label retinal disease classification (45 diseases) with clinical knowledge graph reasoning

---

## 1. Executive Summary

This document outlines the commercialization strategy for a production-grade AI-powered retinal disease screening platform that classifies 45 retinal conditions from fundus images using novel Graph Neural Network architectures with embedded clinical knowledge graphs.

**The core thesis**: Ophthalmology faces a global capacity crisis — there are ~232,000 ophthalmologists serving 2.2 billion people with vision impairment worldwide. Multi-label AI screening that detects 45 conditions simultaneously (not just single-disease like most competitors) can compress a 20-minute specialist review into a 25-millisecond inference, unlocking screening at population scale — especially in Sub-Saharan Africa and South/Southeast Asia where the ophthalmologist-to-population ratio can exceed 1:1,000,000.

**Strategic positioning**: Not a replacement for the ophthalmologist, but a force multiplier — a clinical decision support system (CDSS) that triages, prioritizes, and surfaces the 15-20% of scans that need urgent specialist attention, while safely clearing the 80% that are normal or low-risk.

### Key Numbers

| Metric | Value |
|---|---|
| Total addressable market (TAM) | $5.4B (ophthalmic AI, 2026) |
| Serviceable addressable market (SAM) | $1.8B (multi-disease screening) |
| Serviceable obtainable market (SOM) | $45M (Year 3 target) |
| Unit economics target | $0.30-0.80 per scan (volume-dependent) |
| Regulatory timeline | CE Mark Q4 2026, FDA De Novo Q3 2027 |
| Break-even target | Month 18 post-launch |

---

## 2. Market Analysis: Ophthalmic AI in 2026

### 2.1 Market Size & Growth

The global ophthalmic AI market is valued at approximately **$5.4B in 2026**, growing at **31.2% CAGR** through 2031. Key drivers:

- **Diabetes epidemic**: 537M adults with diabetes globally (IDF 2024); ~35% develop diabetic retinopathy
- **Aging populations**: Age-related macular degeneration affects 196M people globally
- **Screening mandates**: WHO 2025 resolution calling for universal eye screening in primary care
- **Workforce shortage**: Demand for eye exams growing 3x faster than ophthalmologist supply

### 2.2 Market Segmentation

| Segment | Description | Size (2026) | Growth |
|---|---|---|---|
| **Diabetic retinopathy screening** | Single-disease, high-volume primary care | $2.1B | 28% CAGR |
| **Multi-disease screening** | Multi-label detection in specialty clinics | $1.8B | 36% CAGR |
| **Surgical planning AI** | Pre-operative analysis | $0.8B | 25% CAGR |
| **Remote/teleophthalmology** | Cloud-based screening for rural areas | $0.7B | 42% CAGR |

**Our primary segment**: Multi-disease screening ($1.8B) + Remote/teleophthalmology ($0.7B) = **$2.5B combined TAM**.

### 2.3 Buyer Personas

| Persona | Role | Pain Point | Decision Criteria |
|---|---|---|---|
| **Chief Medical Officer** | Clinical champion | Quality assurance, patient safety | Clinical validation, explainability, regulatory status |
| **CIO / CTO** | Technical buyer | Integration complexity, security | API-first architecture, deployment flexibility, SOC2 |
| **Ophthalmologist** | End user | Screening backlog, alert fatigue | Speed, accuracy, referral prioritization, trust |
| **Healthcare Administrator** | Budget holder | Cost per screening, throughput | ROI, reimbursement codes, operational efficiency |
| **Ministry of Health (LMIC)** | Population health buyer | Specialist shortage, rural coverage | Offline capability, cost per scan, language support |

---

## 3. Competitive Landscape & Positioning

### 3.1 Competitor Matrix

| Company | Diseases | Modality | Regulatory | Pricing | Weakness |
|---|---|---|---|---|---|
| **IDx-DR (Digital Diagnostics)** | DR only (binary) | Fundus | FDA cleared (2018) | ~$40/test | Single disease, requires Topcon camera |
| **EyeArt (Eyenuk)** | DR + DME | Fundus | FDA cleared (2020) | $25-35/test | 2 diseases only, no explainability |
| **RetinAI** | DR, AMD, Glaucoma | OCT + Fundus | CE Mark | Enterprise license | OCT-dependent, expensive hardware |
| **Google Health (ARDA)** | DR, DME | Fundus | CE Mark (Thailand) | Research/partnership | Not commercially available, single disease |
| **Visulytix (Pegasus)** | DR, AMD | Fundus | CE Mark (UK) | ~$15/test | 2 diseases, limited geography |
| **Our Platform** | **45 diseases** | Fundus | Pre-conformity | **$0.30-0.80/scan** | Pre-regulatory, dataset scale |

### 3.2 Competitive Moats

Our differentiation rests on five defensible advantages:

**1. Multi-Label Breadth (45 diseases)**
Every competitor screens for 1-3 diseases. We detect 45 simultaneously from a single fundus image. This means one scan replaces what would otherwise require multiple specialist consultations. The clinical knowledge graph encodes 144 disease co-occurrence relationships (e.g., DR often co-presents with HR, BRVO), enabling clinically coherent multi-label predictions.

**2. Clinical Knowledge Graph Reasoning**
Our ClinicalKnowledgeGraph is not just a classifier — it embeds domain knowledge about disease prevalence, severity hierarchies, and co-occurrence patterns directly into inference. This produces clinically plausible predictions, reduces impossible label combinations, and generates referral priority rankings (Emergency / Urgent / Routine).

**3. Four Explainability Methods**
GradCAM, LIME, SHAP, and Integrated Gradients are all production-ready. In 2026, regulatory bodies (FDA, EU AI Act) increasingly require explainability for high-risk AI. Most competitors offer none or one method. We ship four, giving clinicians choice in how they verify AI reasoning.

**4. Cost Structure Advantage**
25ms inference latency means one GPU can serve ~40 scans/second. At $0.30-0.80/scan, we undercut competitors by 30-50x while screening for 15-45x more conditions. This cost advantage is structural — GNN architectures are inherently more parameter-efficient than the large vision transformers competitors use.

**5. LMIC-First Design**
CPU inference support, Docker-based deployment, and clinical knowledge graph calibrated with Ugandan disease prevalence data. This is not an afterthought — the system was designed for deployment in resource-constrained settings from day one.

### 3.3 Positioning Statement

> For ophthalmologists and health systems overwhelmed by screening backlogs, our platform is the only AI clinical decision support system that detects 45 retinal diseases simultaneously from a single fundus image at sub-$1 per scan, with built-in clinical reasoning and four explainability methods — enabling population-scale screening without population-scale ophthalmologist headcount.

---

## 4. Business Model Architecture

### 4.1 Business Model Canvas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BUSINESS MODEL CANVAS                                │
├──────────────┬──────────────┬───────────────┬──────────────┬────────────────┤
│ KEY PARTNERS │ KEY          │ VALUE         │ CUSTOMER     │ CUSTOMER       │
│              │ ACTIVITIES   │ PROPOSITIONS  │ RELATIONS    │ SEGMENTS       │
│              │              │               │              │                │
│ • Fundus     │ • Clinical   │ • 45-disease  │ • Clinical   │ • Hospital     │
│   camera OEMs│   validation │   screening   │   success    │   networks     │
│   (Topcon,   │   studies    │   in one scan │   managers   │   (Tier 1)     │
│   Canon,     │ • Regulatory │ • Sub-$1/scan │ • 24/7       │ • Specialty    │
│   Optomed)   │   submissions│ • Explainable │   clinical   │   eye clinics  │
│ • Regulatory │ • Model      │   AI (4       │   support    │   (Tier 2)     │
│   consultants│   retraining │   methods)    │ • Quarterly  │ • Public       │
│   (FDA, CE)  │ • Customer   │ • Referral    │   model      │   health /     │
│ • CROs for   │   onboarding │   priority    │   performance│   Ministries   │
│   clinical   │ • Platform   │   ranking     │   reviews    │   of Health    │
│   trials     │   ops + SRE  │ • EU AI Act   │ • Training   │   (Tier 3)     │
│ • Cloud      │              │   ready       │   workshops  │ • Telehealth   │
│   (AWS/GCP)  │              │ • CPU + GPU   │              │   platforms    │
│ • Research   │              │   deployment  │              │   (Tier 4)     │
│   hospitals  │              │               │              │                │
├──────────────┴──────────────┼───────────────┼──────────────┴────────────────┤
│ KEY RESOURCES               │               │ CHANNELS                      │
│                             │               │                               │
│ • 4 GNN model architectures │               │ • Direct sales (enterprise)   │
│ • Clinical knowledge graph  │               │ • Camera OEM partnerships     │
│   (144 relationships)       │               │ • Distributor network (LMIC)  │
│ • MLOps infrastructure      │               │ • Medical conferences (AAO,   │
│ • Regulatory dossier        │               │   ARVO, EURETINA, WOC)        │
│ • Clinical validation data  │               │ • Published clinical studies  │
│ • Engineering team          │               │ • Government tenders (LMIC)   │
├─────────────────────────────┴───────────────┴───────────────────────────────┤
│ COST STRUCTURE                              │ REVENUE STREAMS               │
│                                             │                               │
│ Fixed:                                      │ • Per-scan fees ($0.30-0.80)  │
│ • Engineering team (8-12 FTE): $1.2M/yr     │ • Annual platform licenses    │
│ • Regulatory (FDA + CE): $400K one-time     │ • Enterprise SaaS contracts   │
│ • Clinical validation: $300K/study          │ • Government/NGO screening    │
│ • Cloud infrastructure: $180K/yr            │   programs (volume pricing)   │
│ • Regulatory maintenance: $100K/yr          │ • OEM licensing (camera       │
│                                             │   integration)                │
│ Variable:                                   │ • Training & certification    │
│ • GPU compute: ~$0.02/scan (at scale)       │ • Data insights & analytics   │
│ • Customer success: $50K/enterprise/yr      │                               │
│ • Regulatory per-market: $80K-200K          │                               │
└─────────────────────────────────────────────┴───────────────────────────────┘
```

### 4.2 Revenue Model: Tiered Architecture

We employ a **hybrid revenue model** combining per-scan transactional pricing with platform subscription fees, structured across four tiers:

#### Tier 1: Hospital Networks (Enterprise)

| Component | Pricing | Notes |
|---|---|---|
| Platform license | $48,000-120,000/year | Based on # locations |
| Per-scan fee | $0.50-0.80 | Decreasing with volume |
| Implementation | $15,000-30,000 one-time | Integration, training, validation |
| Clinical support | $24,000/year | Dedicated clinical success manager |
| SLA guarantee | Included | p99 < 100ms, 99.9% uptime |

**Target**: 20-50 hospital networks by Year 3
**ACV**: $80,000-$180,000

#### Tier 2: Specialty Eye Clinics (SMB)

| Component | Pricing | Notes |
|---|---|---|
| Monthly subscription | $1,500-3,500/month | Includes up to 2,000 scans |
| Overage per-scan | $0.60 | Above included volume |
| Onboarding | $5,000 one-time | Remote setup + training |

**Target**: 100-300 clinics by Year 3
**ACV**: $24,000-$48,000

#### Tier 3: Public Health / Government Programs (Volume)

| Component | Pricing | Notes |
|---|---|---|
| Per-scan fee | $0.30-0.50 | Volume-based (>100K scans/year) |
| Platform deployment | $25,000-50,000/year | On-premise or sovereign cloud |
| Training program | $10,000/cohort | Train-the-trainer model |

**Target**: 5-15 government programs by Year 3
**ACV**: $60,000-$300,000 (highly variable by population)

#### Tier 4: OEM / Platform Integration

| Component | Pricing | Notes |
|---|---|---|
| API access license | $100,000-500,000/year | White-label or co-branded |
| Per-scan royalty | $0.15-0.30 | Embedded in partner pricing |
| Integration support | $50,000 one-time | SDK, documentation, testing |

**Target**: 3-8 OEM partners by Year 3 (camera manufacturers, telehealth platforms, EHR vendors)

### 4.3 Unit Economics

```
Revenue per scan (blended):                    $0.55
├── GPU compute cost:                         -$0.02
├── Cloud infrastructure (amortized):         -$0.03
├── Customer success (amortized):             -$0.05
├── Regulatory maintenance (amortized):       -$0.02
├── Model retraining (amortized):             -$0.01
└── Gross margin per scan:                     $0.42 (76.4%)

Contribution margin after S&M (amortized):     $0.32 (58.2%)

Break-even volume:    ~4.8M scans/year (or equivalent in platform fees)
Break-even timeline:  Month 18 post-launch (with Tier 1+2 pipeline)
```

---

## 5. Regulatory Strategy

### 5.1 Classification & Pathway

| Jurisdiction | Classification | Pathway | Timeline | Cost |
|---|---|---|---|---|
| **EU (CE Mark)** | Class IIa Medical Device + High-Risk AI | MDR 2017/745 + EU AI Act conformity | Q4 2026 - Q1 2027 | $150K-250K |
| **FDA (USA)** | Class II SaMD (Computer-Aided Detection) | De Novo (no predicate for 45-disease) | Q2 2027 - Q4 2027 | $200K-350K |
| **EAC (East Africa)** | Medical device (varies by country) | National registration (Uganda, Kenya, Rwanda) | Q1 2027 | $30K-60K |
| **India (CDSCO)** | Class B Medical Device Software | MD-9 pathway | Q3 2027 | $40K-80K |
| **UK (MHRA)** | Class IIa (UKCA) | UKCA conformity assessment | Q2 2027 | $80K-120K |

### 5.2 EU AI Act Compliance (August 2026 Enforcement)

Our system is classified as **high-risk AI** under EU AI Act Article 6 (medical devices under MDR). Required compliance measures and our readiness:

| Requirement | Article | Our Status | Evidence |
|---|---|---|---|
| Risk management system | Art. 9 | Implemented | `src/governance/audit_trail.py`, risk documentation |
| Data governance | Art. 10 | Implemented | Dataset cards, data validation pipeline, DVC versioning |
| Technical documentation | Art. 11 | Implemented | Model cards, 12 documentation guides |
| Record-keeping | Art. 12 | Implemented | Prediction logging, audit trail (SHA-256 chained) |
| Transparency | Art. 13 | Implemented | 4 explainability methods, model cards |
| Human oversight | Art. 14 | Implemented | Human-in-the-loop review queue |
| Accuracy, robustness, cybersecurity | Art. 15 | Implemented | 68 tests, drift monitoring, JWT auth, rate limiting |
| Quality management system | Art. 17 | Partial | CI/CD pipeline, needs formal QMS documentation |
| Conformity assessment | Art. 43 | Not started | Requires Notified Body engagement |

**Regulatory advantage**: We are among the first ophthalmic AI systems designed from the ground up for EU AI Act compliance. Most competitors will need to retrofit compliance — we have it architecturally embedded.

### 5.3 FDA Strategy: De Novo Classification

We recommend the **De Novo pathway** because:
- No substantially equivalent predicate exists for 45-disease multi-label retinal screening
- De Novo creates a new classification, establishing us as the predicate for future competitors
- Expected timeline: 12-18 months from submission
- Requires prospective clinical study (300-500 patients, multi-site)

**Pre-submission meeting**: Target Q3 2026 to align on clinical study design and intended use statement.

### 5.4 Clinical Validation Roadmap

| Phase | Study | Sites | Patients | Endpoint | Timeline |
|---|---|---|---|---|---|
| **Retrospective** | Multi-reader multi-case | 3 sites | 1,000 images | Sensitivity/specificity vs. expert panel | Q2-Q3 2026 |
| **Prospective (pilot)** | Single-arm screening | 2 sites (Uganda, UK) | 500 patients | Time-to-diagnosis, referral accuracy | Q4 2026 - Q1 2027 |
| **Prospective (pivotal)** | Multi-site RCT | 5+ sites | 2,000 patients | Sensitivity > 85%, specificity > 80% for top 10 diseases | Q1-Q3 2027 |
| **Real-world evidence** | Registry study | 10+ sites | Ongoing | Performance monitoring, drift detection | Post-launch |

---

## 6. Go-to-Market Strategy

### 6.1 Phased Market Entry

```
PHASE 0: CLINICAL VALIDATION          PHASE 1: EARLY ADOPTERS
(Q2-Q4 2026)                          (Q1-Q2 2027)
┌──────────────────────────┐          ┌──────────────────────────┐
│ • Retrospective study    │          │ • CE Mark obtained       │
│ • EU AI Act conformity   │          │ • 5-10 pilot sites       │
│ • Pre-Sub meeting (FDA)  │          │ • East Africa launch     │
│ • 3 anchor partnerships  │          │ • 2-3 OEM partnerships   │
│ • Hire regulatory + sales│    ──▶   │ • First revenue          │
│ • Seed/Series A ($2-4M)  │          │ • Clinical evidence      │
│                          │          │ • Conference presence     │
│ Revenue: $0              │          │ Revenue: $200K-500K      │
└──────────────────────────┘          └──────────────────────────┘
           │                                     │
           ▼                                     ▼
PHASE 2: SCALE                        PHASE 3: MARKET LEADERSHIP
(Q3 2027 - Q4 2027)                   (2028+)
┌──────────────────────────┐          ┌──────────────────────────┐
│ • FDA clearance          │          │ • 200+ active sites      │
│ • 50-100 active sites    │          │ • 5+ country approvals   │
│ • US market entry        │          │ • OEM embedded in cameras│
│ • Series B ($8-15M)      │          │ • Population screening   │
│ • Expand to 45 diseases  │    ──▶   │   programs (3+ countries)│
│   (start with top 10-15) │          │ • Expanded modalities    │
│ • Government tenders     │          │   (OCT, slit-lamp)       │
│                          │          │ • Platform ecosystem     │
│ Revenue: $2M-5M ARR      │          │ Revenue: $15M-45M ARR    │
└──────────────────────────┘          └──────────────────────────┘
```

### 6.2 Launch Markets (Priority Order)

| Priority | Market | Rationale | Regulatory | Channel |
|---|---|---|---|---|
| **1** | **Uganda + East Africa** | Home market, clinical knowledge graph calibrated for local prevalence, massive specialist shortage (1 ophthalmologist per 400K people), government relationships | EAC registration | Direct + Ministry of Health |
| **2** | **United Kingdom** | Strong NHS AI adoption program (AI Diagnostic Fund), NICE health tech assessment pathway, English-speaking, UKCA pathway | UKCA (Class IIa) | NHS procurement + specialty clinics |
| **3** | **European Union** | CE Mark provides access to 27-country market, EU AI Act readiness is differentiator | CE Mark (MDR) | Distributors + hospital networks |
| **4** | **India** | 1.4B population, 12M ophthalmology visits/year, rapid AI adoption (ABDM digital health infrastructure), cost-sensitive (our strength) | CDSCO Class B | OEM partnerships + Aravind-type eye hospitals |
| **5** | **United States** | Largest revenue market, but most expensive to enter (FDA + sales force) — enter after validation proof points | FDA De Novo | Enterprise sales + EHR integration |

### 6.3 Channel Strategy

**Direct Sales** (Tier 1 & 2): Enterprise sales team targeting hospital networks and large specialty practices. Sales cycle: 3-6 months. Quota-carrying AEs supported by clinical specialists who can demo the explainability features and discuss clinical validation.

**OEM Partnerships** (Tier 4): Embed our API into fundus camera software (Topcon, Canon, Optomed, Remidio) and telehealth platforms. The camera becomes an AI-powered screening station. Revenue share model.

**Government/NGO Channel** (Tier 3): Respond to WHO, World Bank, and national government tenders for population screening programs. Often requires on-premise deployment (our Docker architecture supports this) and local language support.

**Conference-Led Demand Generation**: Presence at AAO, ARVO, EURETINA, WOC, and AIOS. Focus on peer-reviewed publications and live demos. Ophthalmology is a relationship-driven specialty — conference presence converts to pipeline.

### 6.4 Pricing Strategy Rationale

Our pricing is deliberately 30-50x cheaper than incumbents ($0.30-0.80 vs. $25-40/test) for three reasons:

1. **Market creation, not market share**: At $25/test, most of the world cannot afford retinal screening. At $0.50/test, screening becomes viable for primary care clinics in LMICs, school health programs, and diabetic wellness checks. We expand the market rather than fighting for incumbent share.

2. **Volume-driven economics**: Our unit compute cost is $0.02/scan. The margin improves with volume. At 10M scans/year (achievable with 2-3 government programs), our blended cost drops to $0.04/scan.

3. **Lock-in through integration**: Low per-scan pricing encourages high-volume commitments and deep workflow integration. Once we are embedded in a hospital's screening pathway, switching costs are high.

---

## 7. Product Strategy & Roadmap

### 7.1 MVP Feature Prioritization (Phase 0-1)

Not all 45 diseases need to launch simultaneously. Clinical impact and regulatory efficiency demand prioritization:

**Launch Set (10 diseases)** — highest prevalence, strongest clinical evidence, regulatory precedent:

| Disease | Code | Prevalence | Clinical Impact | Regulatory Precedent |
|---|---|---|---|---|
| Diabetic Retinopathy | DR | Very High | Sight-threatening | FDA-cleared predicates exist |
| Age-Related Macular Degeneration | ARMD | Very High | Leading cause of blindness (>50) | Multiple cleared devices |
| Glaucoma (Optic Disc Changes) | ODC/ODP/ODE | High | Irreversible vision loss | Emerging cleared devices |
| Diabetic Macular Edema | DN | High | Requires urgent treatment | FDA-cleared (with DR) |
| Branch Retinal Vein Occlusion | BRVO | Medium | Common vascular emergency | No cleared AI |
| Central Retinal Vein Occlusion | CRVO | Medium | Sight-threatening emergency | No cleared AI |
| Epiretinal Membrane | ERM | Medium | Common surgical indication | No cleared AI |
| Macular Hole | MH | Medium | Surgical emergency | No cleared AI |
| Hypertensive Retinopathy | HR | Medium | Systemic disease marker | No cleared AI |
| Retinitis Pigmentosa | RP | Low | Genetic, early detection critical | No cleared AI |

**Expansion Sets**:
- Phase 2 (Q3 2027): Add 15 more diseases (total 25)
- Phase 3 (2028): Full 45-disease panel

### 7.2 Platform Evolution

```
2026                    2027                    2028                    2029
──────────────────────────────────────────────────────────────────────────────
SCREENING AID           CLINICAL WORKFLOW       PLATFORM                ECOSYSTEM
                        INTEGRATION             EXPANSION

• Single-scan           • EHR integration       • OCT support           • Third-party
  classification          (HL7 FHIR)             (new modality)          model marketplace
• Explainability        • Automated referral    • Longitudinal          • Federated
  dashboard               letters                 progression             learning across
• Referral priority     • Bilateral analysis      tracking                sites
• Model card reports      (both eyes)           • Risk prediction       • Research data
• Audit trail           • Patient timeline        (5-year prognosis)      consortium
                        • Multi-language UI     • Edge deployment       • Digital
                        • Offline mode            (camera-embedded)       therapeutics
                          (edge inference)      • Regulatory dossier      integration
                                                  automation
```

---

## 8. Partnership Strategy

### 8.1 Strategic Partnership Framework

| Partner Type | Target Partners | Value Exchange | Priority |
|---|---|---|---|
| **Camera OEMs** | Optomed (portable), Remidio (smartphone), Canon, Topcon | We get distribution; they get AI differentiation | Critical |
| **Cloud / Infra** | AWS HealthLake, Google Cloud Healthcare, Azure Health | We get credits + compliance infra; they get lighthouse customer | High |
| **Academic / Clinical** | Moorfields Eye Hospital, Aravind Eye Care, Mulago Hospital | We get validation data + clinical credibility; they get AI capability | Critical |
| **EHR Vendors** | Epic, Cerner (Oracle Health), OpenMRS (LMIC) | We get workflow embedding; they get AI module | High |
| **Distributors (LMIC)** | MTN MoMo Health, mPharma, Babyl (Rwanda) | We get last-mile distribution; they get clinical AI content | Medium |
| **Regulatory** | BSI (Notified Body), Emergo (FDA consulting) | We get regulatory navigation; they get a showcase client | High |

### 8.2 Anchor Partnership Targets (Phase 0)

1. **Optomed (Finland)** — Portable fundus camera manufacturer with strong LMIC presence. Our AI + their Aurora camera = an AI-powered portable screening kit. Revenue share on bundled sales.

2. **Moorfields Eye Hospital (London)** — World-leading eye hospital with established AI research program (DeepMind collaboration). Clinical validation partner + entry point into NHS.

3. **Mulago National Referral Hospital (Kampala)** — Largest referral hospital in Uganda. Clinical knowledge graph already calibrated for Ugandan prevalence. Pilot site for East Africa launch.

---

## 9. Financial Projections

### 9.1 Three-Year Revenue Model

| Metric | Year 1 (2027) | Year 2 (2028) | Year 3 (2029) |
|---|---|---|---|
| **Tier 1 (Enterprise)** | 5 customers | 20 customers | 50 customers |
| Tier 1 ACV | $100K | $120K | $150K |
| Tier 1 Revenue | $500K | $2.4M | $7.5M |
| **Tier 2 (Clinics)** | 30 clinics | 120 clinics | 300 clinics |
| Tier 2 ACV | $30K | $36K | $42K |
| Tier 2 Revenue | $900K | $4.3M | $12.6M |
| **Tier 3 (Government)** | 2 programs | 5 programs | 12 programs |
| Tier 3 ACV | $80K | $150K | $250K |
| Tier 3 Revenue | $160K | $750K | $3.0M |
| **Tier 4 (OEM)** | 1 partner | 3 partners | 6 partners |
| Tier 4 ACV | $200K | $300K | $400K |
| Tier 4 Revenue | $200K | $900K | $2.4M |
| **Total Scans** | 2.1M | 9.5M | 32M |
| **Total ARR** | **$1.76M** | **$8.35M** | **$25.5M** |
| **Gross Margin** | 68% | 74% | 78% |

### 9.2 Funding Strategy

| Round | Timing | Amount | Use of Funds | Key Milestones to Unlock |
|---|---|---|---|---|
| **Pre-Seed** (current) | Completed | Bootstrapped | MVP development, initial clinical data | Working product, 45-disease classification |
| **Seed** | Q3 2026 | $2-4M | Regulatory (CE + FDA pre-sub), clinical validation study, first 3 hires (regulatory, sales, clinical) | CE Mark filing, retrospective study results |
| **Series A** | Q2 2027 | $8-15M | FDA submission, US sales team, scale infrastructure, 3 OEM integrations | CE Mark granted, FDA pre-sub complete, first $1M ARR |
| **Series B** | Q1 2029 | $25-40M | International expansion (5+ markets), platform features, 50+ person team | FDA clearance, $8M+ ARR, 3+ government programs |

### 9.3 Investor Profile Targets

- **Seed**: Health-tech focused VCs (Khosla Ventures, 8VC, General Catalyst Health), African health-tech funds (TLcom Capital, Novastar Ventures, Future Africa)
- **Series A**: Growth health-tech VCs (Andreessen Horowitz Bio+Health, GV, Lux Capital) + strategic investors (Optomed, Topcon Ventures)
- **Series B**: Crossover funds + international health investors (IFC, Leapfrog Investments, Axa Venture Partners)

---

## 10. Intellectual Property Strategy

### 10.1 Patent Portfolio Plan

| Patent Application | Category | Scope | Filing |
|---|---|---|---|
| Clinical knowledge graph-augmented GNN for multi-label medical image classification | Core Algorithm | The method of embedding disease co-occurrence, severity, and prevalence priors into graph neural network inference for multi-label diagnosis | Q3 2026 |
| Adaptive multi-resolution encoding with sparse top-K attention for retinal images | Architecture | Multi-resolution patch encoding with O(n*k) attention for efficient fundus analysis | Q4 2026 |
| Automated referral priority ranking from multi-label disease predictions | Clinical Application | Method of computing referral urgency from predicted disease combinations, severities, and clinical relationships | Q1 2027 |
| Drift-aware model governance with immutable audit trail for SaMD | MLOps/Regulatory | System for continuous monitoring, drift detection, and SHA-256 chained audit logging for regulatory compliance | Q1 2027 |

### 10.2 Trade Secrets

- Clinical knowledge graph edge weights and disease relationship encoding (144 relationships)
- Disease prevalence calibration data for specific populations (Uganda, East Africa)
- Optimal per-class threshold configurations for multi-label prediction
- Asymmetric loss hyperparameters tuned for extreme class imbalance in retinal diseases

### 10.3 Defensive IP

- Open-source the base ViGNN architecture (without clinical knowledge graph) to establish prior art and build community
- File provisional patents before any conference publications or preprints
- Consider PCT (Patent Cooperation Treaty) filing for international coverage

---

## 11. Team & Organizational Plan

### 11.1 Founding Team Gaps & Key Hires

| Role | Priority | Why | Timing |
|---|---|---|---|
| **VP Regulatory Affairs** | Critical | CE Mark + FDA pathway execution, QMS establishment | Q3 2026 |
| **Clinical Director (Ophthalmologist)** | Critical | Clinical validation study design, KOL relationships, medical affairs | Q3 2026 |
| **Head of Sales (Enterprise)** | High | Build pipeline for Tier 1 + 2 customers | Q4 2026 |
| **ML Engineer (Production)** | High | Model optimization, ONNX/edge deployment, retraining pipeline | Q3 2026 |
| **DevOps / SRE** | High | Production infrastructure, SOC2, HIPAA compliance | Q4 2026 |
| **Clinical Success Manager** | Medium | Post-sale clinical integration and training | Q1 2027 |
| **Regulatory Associate** | Medium | Documentation, submission preparation, post-market surveillance | Q1 2027 |

### 11.2 Organizational Structure (Year 2)

```
CEO / Co-Founder
├── CTO / Co-Founder
│   ├── ML Engineering (3)
│   ├── Platform Engineering (2)
│   └── DevOps / SRE (1)
├── VP Regulatory Affairs
│   ├── Regulatory Associate (1)
│   └── QA / QMS (1)
├── Clinical Director
│   ├── Clinical Success (2)
│   └── Medical Affairs (1)
├── Head of Sales
│   ├── Enterprise AEs (2)
│   └── Partnerships (1)
└── Head of Operations
    ├── Finance (1)
    └── Legal (outsourced)

Total: ~18 FTE by end of Year 2
```

---

## 12. Risk Framework

### 12.1 Risk Matrix

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **FDA De Novo rejection or delay** | Medium | High | Engage regulatory consultant early, align on clinical study design via Pre-Sub, prepare 510(k) fallback (subset of diseases) |
| **Clinical validation fails to meet endpoints** | Low-Medium | Critical | Power study adequately, use adaptive trial design, focus on top 10 diseases first |
| **Competitor launches 10+ disease screening** | Medium | Medium | Accelerate to market, leverage LMIC-first positioning, deepen clinical knowledge graph moat |
| **Dataset bias (demographics)** | Medium | High | Multi-site validation, fairness evaluation framework already built, diversify training data through clinical partnerships |
| **GPU cost increase / supply constraints** | Low | Medium | CPU inference support already built, ONNX export enables diverse hardware, negotiate cloud commitments |
| **Regulatory landscape changes** | Medium | Medium | EU AI Act compliance already embedded, maintain regulatory counsel, participate in standards bodies |
| **Reimbursement challenges (CPT codes)** | High (US) | High (US) | Start with non-reimbursement markets (LMIC, self-pay), apply for CPT Category III code, partner with payers |
| **Key person dependency** | Medium | High | Document all systems (12 guides already written), cross-train team, vest founders over 4 years |
| **Cybersecurity / data breach** | Low | Critical | JWT auth, rate limiting, security scanning in CI, SOC2 Type II by Year 2, HIPAA BAA for US customers |

### 12.2 Kill Criteria

The venture should be reconsidered if any of the following occur:

1. Retrospective clinical study shows sensitivity < 70% for top 5 diseases (indicates fundamental model limitation)
2. CE Mark and FDA both rejected after appeals (regulatory path blocked)
3. Three or more competitors achieve 15+ disease screening with regulatory approval before our first clearance
4. Unable to raise Seed round by Q1 2027 (market does not believe in the opportunity)
5. Zero paying customers after 12 months post-CE Mark (product-market fit failure)

---

## 13. Implementation Roadmap: First 18 Months

### Q2 2026 (Now - June 2026)

- [ ] Complete retrospective clinical validation study design
- [ ] Engage regulatory consultant for CE Mark + FDA Pre-Sub
- [ ] File first provisional patent (clinical knowledge graph GNN)
- [ ] Begin Seed fundraising ($2-4M target)
- [ ] Establish advisory board (2 ophthalmologists, 1 regulatory expert, 1 health-tech founder)
- [ ] Sign LOI with Mulago Hospital for pilot site

### Q3 2026

- [ ] Hire VP Regulatory Affairs + Clinical Director + ML Engineer
- [ ] Execute retrospective study (1,000 images, 3 sites)
- [ ] Submit CE Mark technical documentation to Notified Body
- [ ] FDA Pre-Submission meeting
- [ ] Close Seed round
- [ ] Begin Optomed partnership discussions
- [ ] SOC2 Type I preparation

### Q4 2026

- [ ] EU AI Act conformity assessment (August 2026 enforcement deadline)
- [ ] Prospective pilot study (500 patients, Uganda + UK)
- [ ] CE Mark expected grant
- [ ] Hire Head of Sales
- [ ] First pilot deployments (Tier 2, 3-5 clinics)
- [ ] Conference presence: AAO Annual Meeting

### Q1 2027

- [ ] First revenue (Tier 2 clinics + Tier 3 pilot)
- [ ] East Africa market launch (Uganda, Kenya, Rwanda)
- [ ] FDA De Novo submission
- [ ] EHR integration development (HL7 FHIR)
- [ ] Second patent filing (referral priority ranking)
- [ ] Hire clinical success team

### Q2 2027

- [ ] Series A fundraising ($8-15M)
- [ ] UK market entry (UKCA + NHS AI Diagnostic Fund application)
- [ ] First OEM partnership signed (Optomed or Remidio)
- [ ] 20+ active clinical sites
- [ ] Pivotal clinical study initiation (2,000 patients, 5+ sites)
- [ ] Conference presence: ARVO Annual Meeting

### Q3-Q4 2027

- [ ] FDA clearance (target)
- [ ] US market entry preparation
- [ ] India CDSCO submission
- [ ] $2M+ ARR milestone
- [ ] 50+ active clinical sites
- [ ] Platform v2: bilateral analysis, patient timeline, multi-language

---

## 14. Success Metrics & KPIs

### Clinical

| Metric | Year 1 Target | Year 3 Target |
|---|---|---|
| Sensitivity (top 10 diseases) | > 85% | > 90% |
| Specificity (top 10 diseases) | > 80% | > 85% |
| Referral accuracy | > 90% | > 95% |
| Time-to-diagnosis reduction | 40% | 60% |
| False positive rate | < 15% | < 10% |

### Commercial

| Metric | Year 1 Target | Year 3 Target |
|---|---|---|
| ARR | $1.76M | $25.5M |
| Total scans processed | 2.1M | 32M |
| Active clinical sites | 38 | 362 |
| Net revenue retention | 110% | 130% |
| Customer acquisition cost (CAC) | $15K | $10K |
| CAC payback period | 8 months | 5 months |
| Gross margin | 68% | 78% |

### Operational

| Metric | Target |
|---|---|
| System uptime | 99.9% |
| Inference latency (p99) | < 100ms |
| Drift detection response time | < 24 hours |
| Model retraining cadence | Quarterly |
| Regulatory audit readiness | Always |

---

## 15. Strategic Summary

### Why This Wins

1. **Right problem, right time**: 2.2B people with vision impairment, <250K ophthalmologists globally, EU AI Act creating regulatory moats for compliant systems, AI reimbursement codes emerging.

2. **Structural cost advantage**: 25ms inference = $0.02/scan compute cost. Competitors charging $25-40/test are 50-100x our cost basis. We can profitably serve markets they cannot.

3. **Technical moat**: Clinical knowledge graph with 144 disease relationships is not a model weight — it is structured clinical knowledge that takes years of ophthalmological expertise to build and validate. Combined with 45-disease multi-label classification, this is defensible.

4. **LMIC-first is a strategy, not charity**: East Africa has the worst ophthalmologist shortage and the least AI competition. We build clinical evidence, refine the product, and generate revenue in markets where we are the only option — then enter the US/EU with validated technology and real-world evidence that no competitor has.

5. **Regulatory-first architecture**: EU AI Act compliance is baked in, not bolted on. As regulations tighten globally, our head start becomes a compounding advantage.

### The Ask

$2-4M Seed round to execute Phase 0 (clinical validation + regulatory submission) and Phase 1 (first revenue), positioning for a $8-15M Series A at CE Mark + first $1M ARR milestone.

---

*This strategy is a living document. Review quarterly and update based on clinical validation results, regulatory feedback, and market signals.*
