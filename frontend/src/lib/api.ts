// Use same-origin paths so the Next rewrite + httpOnly cookies work seamlessly.
// In production set BACKEND_URL on the Next runtime; in dev defaults to localhost:8080.
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

// ── System Info ──
export interface SystemInfo {
  platform: {
    name: string;
    version: string;
    environment: string;
    region: string;
    regulatory_mode: string;
  };
  model: {
    name: string;
    loaded: boolean;
    num_classes: number;
    diseases_covered: number;
    knowledge_graph_edges: number;
    threshold_source: string;
  };
  infrastructure: {
    python_version: string;
    pytorch_version: string;
    cuda_available: boolean;
    cuda_version: string | null;
    gpu: string | null;
    gpu_memory: string | null;
    device: string;
  };
  capabilities: {
    explainability_methods: string[];
    clinical_reasoning: boolean;
    knowledge_graph: boolean;
    human_review: boolean;
    audit_trail: boolean;
    drift_detection: boolean;
  };
  compliance: {
    eu_ai_act: string;
    fda_samd: string;
    data_governance: boolean;
    model_cards: boolean;
    fairness_evaluation: boolean;
    prediction_logging: boolean;
  };
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const res = await fetch(`${API_BASE}/api/v1/system/info`, { credentials: "include" });
  if (!res.ok) throw new Error("System info unavailable");
  return res.json();
}

// ── Analytics ──
export interface AnalyticsSummary {
  total_scans: number;
  today_scans: number;
  avg_inference_ms: number;
  referral_distribution: Record<string, number>;
  top_detected_diseases: { code: string; count: number }[];
  daily_volumes: { date: string; scans: number }[];
}

export async function fetchAnalytics(): Promise<AnalyticsSummary> {
  const res = await fetch(`${API_BASE}/api/v1/analytics/summary`, { credentials: "include" });
  if (!res.ok) throw new Error("Analytics unavailable");
  return res.json();
}

// ── All Disease Info ──
export interface AllDiseaseInfo {
  total: number;
  diseases: (DiseaseInfo & { info_available: boolean })[];
}

export async function fetchAllDiseaseInfo(): Promise<AllDiseaseInfo> {
  const res = await fetch(`${API_BASE}/api/v1/clinical/disease-info`, { credentials: "include" });
  if (!res.ok) throw new Error("Disease info unavailable");
  return res.json();
}

// ── Review Queue ──
export interface ReviewItem {
  request_id: string;
  prediction_id: string;
  reason: string;
  priority: string;
  created_at: string;
  summary: Record<string, unknown>;
}

export interface PendingReviews {
  total_pending: number;
  reviews: ReviewItem[];
}

export async function fetchPendingReviews(priority?: string): Promise<PendingReviews> {
  const url = priority
    ? `${API_BASE}/api/v1/review/pending?priority=${priority}`
    : `${API_BASE}/api/v1/review/pending`;
  const res = await fetch(url, { credentials: "include" });
  if (!res.ok) throw new Error("Reviews unavailable");
  return res.json();
}

export interface ReviewStats {
  total: number;
  pending: number;
  resolved: number;
  by_priority: Record<string, number>;
}

export async function fetchReviewStats(): Promise<ReviewStats> {
  const res = await fetch(`${API_BASE}/api/v1/review/stats`, { credentials: "include" });
  if (!res.ok) throw new Error("Review stats unavailable");
  return res.json();
}

export async function resolveReview(
  requestId: string,
  decision: "confirmed" | "rejected",
  reviewer?: string,
  notes?: string,
): Promise<{ status: string; request_id: string; decision: string }> {
  const res = await fetch(`${API_BASE}/api/v1/review/${requestId}/resolve`, {
    credentials: "include",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      reviewer: reviewer ?? "clinical_user",
      decision,
      notes: notes ?? "",
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Failed to resolve review");
  }
  return res.json();
}

// ── Agents ──
export interface AgentStatus {
  total_agents: number;
  agents: Record<string, {
    name: string;
    status: string;
    last_action: string;
    last_action_at: string;
    actions_taken: number;
    errors: number;
    started_at: string;
    tools: string[];
  }>;
  event_bus: {
    total_events: number;
    subscriber_count: number;
    event_types_seen: Record<string, number>;
  };
}

export async function fetchAgentStatus(): Promise<AgentStatus> {
  const res = await fetch(`${API_BASE}/api/v1/agents/status`, { credentials: "include" });
  if (!res.ok) throw new Error("Agent status unavailable");
  return res.json();
}

export interface AgentEvent {
  event_id: string;
  type: string;
  source: string;
  timestamp: string;
  data: Record<string, unknown>;
}

export async function fetchAgentEvents(limit = 20): Promise<{ total: number; events: AgentEvent[] }> {
  const res = await fetch(`${API_BASE}/api/v1/agents/events?limit=${limit}`, { credentials: "include" });
  if (!res.ok) throw new Error("Agent events unavailable");
  return res.json();
}

export async function fetchComplianceReport(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/api/v1/agents/compliance`, { credentials: "include" });
  if (!res.ok) throw new Error("Compliance report unavailable");
  return res.json();
}

// ── Health ──
export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`, { credentials: "include" });
  if (!res.ok) throw new Error("API unreachable");
  return res.json() as Promise<{
    status: string;
    model_loaded: boolean;
    device: string;
    diseases_count: number;
  }>;
}

export async function fetchModelHealth() {
  const res = await fetch(`${API_BASE}/health/model`, { credentials: "include" });
  if (!res.ok) throw new Error("Model health unavailable");
  return res.json() as Promise<{
    latency_p50_ms: number;
    latency_p95_ms: number;
    latency_p99_ms: number;
    throughput_rps: number;
    error_rate: number;
    total_predictions: number;
    uptime_seconds: number;
    sla_compliant: boolean;
  }>;
}

// ── Diseases ──
export async function fetchDiseases() {
  const res = await fetch(`${API_BASE}/api/v1/diseases`, { credentials: "include" });
  if (!res.ok) throw new Error("Failed to fetch diseases");
  return res.json() as Promise<{
    total: number;
    diseases: { code: string; name: string }[];
  }>;
}

// ── Knowledge Graph ──
export interface KnowledgeGraphData {
  diseases: number;
  edges: number;
  categories: Record<string, string[]>;
  relationships: { source: string; target: string; type: string }[];
  severity: Record<string, number>;
  prevalence: Record<string, number>;
  disease_names: Record<string, string>;
}

export async function fetchKnowledgeGraph(): Promise<KnowledgeGraphData> {
  const res = await fetch(`${API_BASE}/api/v1/clinical/knowledge-graph`, { credentials: "include" });
  if (!res.ok) throw new Error("Knowledge graph unavailable");
  return res.json();
}

// ── Disease Info ──
export interface DiseaseInfo {
  code: string;
  name: string;
  info_available: boolean;
  severity?: number;
  category?: string;
  description?: string;
  risk_factors?: string[];
  treatment?: string[];
  urgency?: string;
}

export async function fetchDiseaseInfo(code: string): Promise<DiseaseInfo> {
  const res = await fetch(`${API_BASE}/api/v1/clinical/disease-info/${code}`, { credentials: "include" });
  if (!res.ok) throw new Error("Disease info unavailable");
  return res.json();
}

// ── Prediction ──
export interface Prediction {
  code: string;
  name: string;
  probability: number;
  threshold?: number;
  confidence: "high" | "medium" | "low";
}

export interface PredictionResponse {
  success: boolean;
  predictions: Prediction[];
  total_detected: number;
  all_probabilities: Record<string, { probability: number; name: string; threshold?: number }>;
  clinical: {
    referral_priority: string;
    refined_predictions: Record<string, number>;
  };
  inference_ms: number;
  model_loaded: boolean;
  threshold: number;
  threshold_source?: string;
  per_class_thresholds?: Record<string, number>;
  fundus_gate?: { passed: boolean; confidence: number };
  ood_warning?: { flagged: boolean; message: string; checks: Record<string, unknown> };
}

export async function predictImage(file: File, threshold = 0.5): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/v1/predict?threshold=${threshold}`, {
    method: "POST",
    body: form,
    credentials: "include",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    // Handle structured error from fundus gate (422)
    const detail = err.detail;
    if (typeof detail === "object" && detail?.error === "non_fundus_image") {
      throw new Error(detail.message || "Image does not appear to be a retinal fundus photograph.");
    }
    throw new Error(typeof detail === "string" ? detail : "Prediction failed");
  }
  return res.json();
}

// ── Clinical Reasoning ──
export interface ReasoningResponse {
  adjustments: {
    disease: string;
    name: string;
    original: number;
    refined: number;
    boost: number;
    reason: string;
  }[];
  referral_priority: string;
  visual_findings: Record<string, number>;
  treatment_recommendations: Record<string, string[]>;
  detected_count: number;
}

export async function explainReasoning(
  predictions: Record<string, number>,
): Promise<ReasoningResponse> {
  const res = await fetch(`${API_BASE}/api/v1/clinical/explain-reasoning`, {
    credentials: "include",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(predictions),
  });
  if (!res.ok) throw new Error("Reasoning unavailable");
  return res.json();
}

// ── Explainability ──
export interface GradCAMHeatmap {
  class_index: number;
  disease: string;
  disease_name: string;
  probability: number;
  heatmap: string | null; // base64 data URI
  error?: string;
}

export interface GradCAMResponse {
  method: string;
  original: string; // base64
  heatmaps: GradCAMHeatmap[];
  elapsed_ms: number;
}

export interface LIMEExplanation {
  prediction: number;
  segments: number;
  samples_used: number;
  summary: {
    top_positive_features: number;
    top_negative_features: number;
    max_weight: number;
    min_weight: number;
  };
  feature_weights: Record<string, number>;
  error?: string;
}

export interface LIMEResponse {
  method: string;
  explanations: Record<string, LIMEExplanation>;
  elapsed_ms: number;
}

export interface SHAPExplanation {
  prediction: number;
  feature_importance: {
    mean_abs_shap: number;
    max_abs_shap: number;
    std_shap: number;
  };
  error?: string;
}

export interface SHAPResponse {
  method: string;
  explanations: Record<string, SHAPExplanation>;
  elapsed_ms: number;
}

export interface IGExplanation {
  attribution_summary: {
    mean: number;
    max: number;
    min: number;
  };
  prediction: number;
}

export interface IGResponse {
  method: string;
  explanations: Record<string, IGExplanation>;
  elapsed_ms: number;
}

export interface ELI5Explanation {
  prediction: number;
  confidence_level: string;
  feature_importance: Record<string, number>;
  explanation_text: string;
  top_contributing_features: {
    feature: string;
    weight: number;
    direction: "positive" | "negative";
  }[];
  eli5_summary: {
    model_type: string;
    explanation_method: string;
    feature_count: number;
    prediction_threshold: number;
  };
  error?: string;
}

export interface ELI5Response {
  method: string;
  explanations: Record<string, ELI5Explanation>;
  elapsed_ms: number;
}

export interface AvailableMethods {
  model_loaded: boolean;
  methods: Record<string, { available: boolean; description: string }>;
}

export async function fetchAvailableMethods(): Promise<AvailableMethods> {
  const res = await fetch(`${API_BASE}/api/v1/explain/available`, { credentials: "include" });
  if (!res.ok) throw new Error("Methods unavailable");
  return res.json();
}

export async function explainGradCAM(file: File, topK = 3): Promise<GradCAMResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/v1/explain/gradcam?top_k=${topK}`, {
    credentials: "include",
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "GradCAM failed");
  }
  return res.json();
}

export async function explainLIME(file: File, topK = 3, numSamples = 300): Promise<LIMEResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(
    `${API_BASE}/api/v1/explain/lime?top_k=${topK}&num_samples=${numSamples}`,
    { credentials: "include", method: "POST", body: form },
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "LIME failed");
  }
  return res.json();
}

export async function explainSHAP(file: File, topK = 3): Promise<SHAPResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/v1/explain/shap?top_k=${topK}`, {
    credentials: "include",
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "SHAP failed");
  }
  return res.json();
}

export async function explainIG(file: File, topK = 2, nSteps = 25): Promise<IGResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(
    `${API_BASE}/api/v1/explain/integrated-gradients?top_k=${topK}&n_steps=${nSteps}`,
    { credentials: "include", method: "POST", body: form },
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Integrated Gradients failed");
  }
  return res.json();
}

export async function explainELI5(file: File, topK = 3): Promise<ELI5Response> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/v1/explain/eli5?top_k=${topK}`, {
    credentials: "include",
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "ELI5 failed");
  }
  return res.json();
}
