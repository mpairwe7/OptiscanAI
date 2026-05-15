/**
 * Plan catalog — kept in sync with backend `plans` table seed.
 *
 * The backend is the source of truth at runtime (see GET /api/v1/billing/plans);
 * this file is the *fallback* shape used at build/server-render time so the
 * marketing page renders even when the API is unreachable.
 */

export type PlanId = "free" | "clinician" | "practice" | "health_system";
export type BillingPeriod = "monthly" | "annual";

export interface FeatureEntry {
  key: string;
  label: string;
  /** Free | Clinician | Practice | Health System */
  byPlan: [string, string, string, string];
}

export interface Plan {
  id: PlanId;
  name: string;
  tagline: string;
  description: string;
  priceUsd: { monthly: number | "contact"; annual: number | "contact" };
  scanQuota: number | "unlimited";
  seats: number | "unlimited";
  cta: { label: string; href: string; variant: "primary" | "outline" };
  highlight?: boolean;
}

export const ANNUAL_DISCOUNT_PCT = 0.17;

export const PLANS: Plan[] = [
  {
    id: "free",
    name: "Free",
    tagline: "Try clinical screening with no commitment.",
    description: "10 scans/month, Grad-CAM, community support.",
    priceUsd: { monthly: 0, annual: 0 },
    scanQuota: 10,
    seats: 1,
    cta: { label: "Get started", href: "/sign-up?plan=free", variant: "outline" },
  },
  {
    id: "clinician",
    name: "Clinician",
    tagline: "For solo ophthalmologists and optometrists.",
    description: "500 scans/month, all explainability methods, PDF reports.",
    priceUsd: { monthly: 29, annual: 290 },
    scanQuota: 500,
    seats: 1,
    highlight: true,
    cta: { label: "Subscribe", href: "/app/checkout/clinician", variant: "primary" },
  },
  {
    id: "practice",
    name: "Practice",
    tagline: "For multi-clinician clinics.",
    description: "3,000 scans/month, 5 seats, audit log, review queue.",
    priceUsd: { monthly: 149, annual: 1490 },
    scanQuota: 3000,
    seats: 5,
    cta: { label: "Subscribe", href: "/app/checkout/practice", variant: "primary" },
  },
  {
    id: "health_system",
    name: "Health System",
    tagline: "For hospitals and regional health offices.",
    description: "Unlimited scans, SSO/SCIM, BAA + PDP, DHIS2/FHIR/DICOM.",
    priceUsd: { monthly: "contact", annual: "contact" },
    scanQuota: "unlimited",
    seats: "unlimited",
    cta: { label: "Contact sales", href: "/contact-sales", variant: "outline" },
  },
];

export const FEATURE_MATRIX: FeatureEntry[] = [
  { key: "scans", label: "Scans per month", byPlan: ["10", "500", "3,000", "Unlimited"] },
  { key: "diseases", label: "Disease detection", byPlan: ["45 diseases", "45 diseases", "45 diseases", "45 diseases"] },
  { key: "grad_cam", label: "Grad-CAM heatmaps", byPlan: ["✓", "✓", "✓", "✓"] },
  { key: "advanced_xai", label: "LIME, SHAP, Integrated Gradients, ELI5", byPlan: ["—", "✓", "✓", "✓"] },
  { key: "clinical_reasoning", label: "Clinical knowledge-graph reasoning", byPlan: ["—", "✓", "✓", "✓"] },
  { key: "voice_mode", label: "Voice-first mode", byPlan: ["—", "✓", "✓", "✓"] },
  { key: "pdf", label: "PDF report export", byPlan: ["Watermarked", "✓", "✓", "✓"] },
  { key: "retention", label: "Report retention", byPlan: ["7 days", "1 year", "3 years", "Custom"] },
  { key: "review_queue", label: "Multi-clinician review queue", byPlan: ["—", "—", "✓", "✓"] },
  { key: "audit_log", label: "Audit log + CSV export", byPlan: ["—", "—", "✓", "✓"] },
  { key: "fairness", label: "Fairness dashboard", byPlan: ["—", "—", "✓", "✓"] },
  { key: "team_seats", label: "Team seats included", byPlan: ["1", "1", "5", "Unlimited"] },
  { key: "dhis2_fhir", label: "DHIS2 + FHIR integrations", byPlan: ["—", "—", "—", "✓"] },
  { key: "dicom", label: "DICOM upload", byPlan: ["—", "—", "—", "✓"] },
  { key: "sso", label: "SSO + SCIM", byPlan: ["—", "—", "—", "✓"] },
  { key: "baa", label: "BAA / PDP Act compliance", byPlan: ["—", "—", "—", "✓"] },
  { key: "support", label: "Support", byPlan: ["Community", "Email", "Priority email + chat", "Dedicated CSM + SLA"] },
];

/**
 * Maps a locked feature to the cheapest plan that unlocks it.
 * Used by UpsellSheet — clicking a locked tab pre-selects the right upgrade.
 */
export const FEATURE_UNLOCKED_BY: Record<string, PlanId> = {
  lime: "clinician",
  shap: "clinician",
  integrated_gradients: "clinician",
  eli5: "clinician",
  clinical_reasoning: "clinician",
  voice_mode: "clinician",
  review_queue: "practice",
  audit_log: "practice",
  team_seats: "practice",
  dhis2: "health_system",
  fhir: "health_system",
  dicom: "health_system",
  sms_referral: "health_system",
  sso: "health_system",
};

export function planById(id: PlanId): Plan | undefined {
  return PLANS.find((p) => p.id === id);
}

export function priceFor(plan: Plan, period: BillingPeriod): number | "contact" {
  return plan.priceUsd[period];
}

export function formatPrice(plan: Plan, period: BillingPeriod): string {
  const p = priceFor(plan, period);
  if (p === "contact") return "Contact sales";
  if (p === 0) return "$0";
  if (period === "annual") {
    const monthly = Math.round((p / 12) * 100) / 100;
    return `$${monthly.toFixed(0)}/mo`;
  }
  return `$${p}/mo`;
}

export function annualSavingsLabel(plan: Plan): string | null {
  const m = plan.priceUsd.monthly;
  const a = plan.priceUsd.annual;
  if (m === "contact" || a === "contact" || m === 0) return null;
  const saved = m * 12 - a;
  if (saved <= 0) return null;
  return `Save $${saved}/yr`;
}
