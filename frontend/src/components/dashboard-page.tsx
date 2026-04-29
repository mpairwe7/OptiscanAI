"use client";
import { useQuery } from "@tanstack/react-query";
import { fetchHealth, fetchModelHealth, fetchAnalytics, fetchSystemInfo } from "@/lib/api";
import { useAppStore } from "@/stores/app-store";

function StatCard({ label, value, unit, trend, color = "teal" }: {
  label: string; value: string | number; unit?: string; trend?: string; color?: string;
}) {
  const colors: Record<string, string> = {
    teal: "border-teal-200 bg-teal-50/50",
    blue: "border-blue-200 bg-blue-50/50",
    amber: "border-amber-200 bg-amber-50/50",
    red: "border-red-200 bg-red-50/50",
    emerald: "border-emerald-200 bg-emerald-50/50",
  };
  return (
    <div className={`rounded-xl border p-3 sm:p-4 ${colors[color] ?? colors.teal}`}>
      <div className="text-[11px] sm:text-xs font-medium text-slate-500 mb-1">{label}</div>
      <div className="flex items-baseline gap-1">
        <span className="text-xl sm:text-2xl font-bold text-slate-800">{value}</span>
        {unit && <span className="text-[10px] sm:text-xs text-slate-400">{unit}</span>}
      </div>
      {trend && <div className="text-[10px] sm:text-xs text-emerald-600 mt-1">{trend}</div>}
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="rounded-xl border border-slate-200 p-3 sm:p-4 space-y-2">
      <div className="skeleton h-3 w-20" />
      <div className="skeleton h-7 w-16" />
    </div>
  );
}

function ComplianceBadge({ label, status }: { label: string; status: string }) {
  const isReady = status === "conformity_ready" || status === "true" || status === true as unknown as string;
  return (
    <div className="flex items-center gap-2 py-1.5">
      <span className={`w-2 h-2 rounded-full shrink-0 ${isReady ? "bg-emerald-400" : "bg-amber-400"}`} />
      <span className="text-xs sm:text-sm text-slate-600 flex-1">{label}</span>
      <span className={`text-[10px] sm:text-xs font-medium px-2 py-0.5 rounded-full ${
        isReady ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"
      }`}>
        {typeof status === "boolean" ? (status ? "Active" : "Inactive") : status.replace(/_/g, " ")}
      </span>
    </div>
  );
}

export function DashboardPage() {
  const { setPage, scanHistory } = useAppStore();
  const health = useQuery({ queryKey: ["health"], queryFn: fetchHealth, refetchInterval: 10_000 });
  const modelHealth = useQuery({ queryKey: ["model-health"], queryFn: fetchModelHealth, refetchInterval: 15_000 });
  const analytics = useQuery({ queryKey: ["analytics"], queryFn: fetchAnalytics, refetchInterval: 30_000 });
  const sysInfo = useQuery({ queryKey: ["system-info"], queryFn: fetchSystemInfo, staleTime: 60_000 });

  return (
    <div className="space-y-4 sm:space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-800">Dashboard</h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-0.5">
            Clinical screening platform overview
          </p>
        </div>
        <button
          onClick={() => setPage("screening")}
          className="px-4 py-2.5 bg-teal-600 text-white rounded-lg font-semibold text-sm
                     hover:bg-teal-700 active:bg-teal-800 transition-colors shadow-sm w-full sm:w-auto"
        >
          New Screening
        </button>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        {analytics.isLoading ? (
          <>
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </>
        ) : (
          <>
            <StatCard label="Total Scans" value={analytics.data?.total_scans ?? 0} color="teal" />
            <StatCard label="Today" value={analytics.data?.today_scans ?? 0} unit="scans" color="blue" />
            <StatCard label="Avg Inference" value={analytics.data?.avg_inference_ms?.toFixed(1) ?? "0"} unit="ms" color="emerald" />
            <StatCard label="Diseases" value={health.data?.diseases_count ?? 45} color="amber" />
          </>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
        {/* Model Performance */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100 flex items-center justify-between">
            <h2 className="font-semibold text-sm sm:text-base text-slate-800">Model Performance</h2>
            {modelHealth.data && (
              <span className={`text-[10px] sm:text-xs px-2 sm:px-2.5 py-0.5 sm:py-1 rounded-full font-medium ${
                modelHealth.data.sla_compliant
                  ? "bg-emerald-100 text-emerald-700"
                  : "bg-red-100 text-red-700"
              }`}>
                SLA {modelHealth.data.sla_compliant ? "OK" : "Violated"}
              </span>
            )}
          </div>
          <div className="p-4 sm:p-5">
            {modelHealth.isLoading ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="text-center space-y-2">
                    <div className="skeleton h-8 w-20 mx-auto" />
                    <div className="skeleton h-3 w-16 mx-auto" />
                  </div>
                ))}
              </div>
            ) : modelHealth.data ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6">
                <div className="text-center">
                  <div className="text-xl sm:text-2xl font-bold text-slate-800">
                    {modelHealth.data.latency_p50_ms.toFixed(1)}
                    <span className="text-[10px] sm:text-xs font-normal text-slate-400 ml-0.5">ms</span>
                  </div>
                  <div className="text-[10px] sm:text-xs text-slate-500 mt-1">Latency P50</div>
                </div>
                <div className="text-center">
                  <div className="text-xl sm:text-2xl font-bold text-slate-800">
                    {modelHealth.data.latency_p95_ms.toFixed(1)}
                    <span className="text-[10px] sm:text-xs font-normal text-slate-400 ml-0.5">ms</span>
                  </div>
                  <div className="text-[10px] sm:text-xs text-slate-500 mt-1">Latency P95</div>
                </div>
                <div className="text-center">
                  <div className="text-xl sm:text-2xl font-bold text-slate-800">
                    {modelHealth.data.throughput_rps.toFixed(1)}
                    <span className="text-[10px] sm:text-xs font-normal text-slate-400 ml-0.5">rps</span>
                  </div>
                  <div className="text-[10px] sm:text-xs text-slate-500 mt-1">Throughput</div>
                </div>
                <div className="text-center">
                  <div className={`text-xl sm:text-2xl font-bold ${
                    modelHealth.data.error_rate > 0.01 ? "text-red-600" : "text-emerald-600"
                  }`}>
                    {(modelHealth.data.error_rate * 100).toFixed(2)}%
                  </div>
                  <div className="text-[10px] sm:text-xs text-slate-500 mt-1">Error Rate</div>
                </div>
              </div>
            ) : (
              <div className="text-sm text-slate-400 text-center py-6">Loading metrics...</div>
            )}

            {/* Referral Distribution */}
            {analytics.data?.referral_distribution && Object.keys(analytics.data.referral_distribution).length > 0 && (
              <div className="mt-4 sm:mt-6 pt-4 sm:pt-5 border-t border-slate-100">
                <h3 className="text-[10px] sm:text-xs font-semibold text-slate-500 uppercase mb-2 sm:mb-3">Referral Distribution</h3>
                <div className="flex flex-col sm:flex-row gap-2 sm:gap-3">
                  {Object.entries(analytics.data.referral_distribution).map(([priority, count]) => {
                    const total = Object.values(analytics.data!.referral_distribution).reduce((a, b) => a + b, 0);
                    const pct = total > 0 ? Math.round((count / total) * 100) : 0;
                    const colors: Record<string, string> = {
                      URGENT: "bg-red-500", ROUTINE: "bg-amber-500", FOLLOW_UP: "bg-emerald-500",
                      EMERGENCY: "bg-red-700",
                    };
                    return (
                      <div key={priority} className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-[10px] sm:text-xs text-slate-600">{priority}</span>
                          <span className="text-[10px] sm:text-xs font-mono text-slate-400">{pct}%</span>
                        </div>
                        <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                          <div className={`h-full rounded-full ${colors[priority] ?? "bg-slate-400"}`}
                               style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Compliance & Capabilities */}
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100">
            <h2 className="font-semibold text-sm sm:text-base text-slate-800">Compliance Status</h2>
          </div>
          <div className="p-4 sm:p-5 space-y-1">
            {sysInfo.isLoading ? (
              <div className="space-y-3">
                {[1, 2, 3, 4].map((i) => <div key={i} className="skeleton h-6 w-full" />)}
              </div>
            ) : sysInfo.data ? (
              <>
                <ComplianceBadge label="EU AI Act" status={sysInfo.data.compliance.eu_ai_act} />
                <ComplianceBadge label="FDA SaMD" status={sysInfo.data.compliance.fda_samd} />
                <ComplianceBadge label="Data Governance" status={String(sysInfo.data.compliance.data_governance)} />
                <ComplianceBadge label="Model Cards" status={String(sysInfo.data.compliance.model_cards)} />
                <ComplianceBadge label="Fairness Eval" status={String(sysInfo.data.compliance.fairness_evaluation)} />
                <ComplianceBadge label="Prediction Logging" status={String(sysInfo.data.compliance.prediction_logging)} />
              </>
            ) : (
              <div className="text-sm text-slate-400 py-4 text-center">Loading...</div>
            )}
          </div>

          {sysInfo.data && (
            <div className="px-4 sm:px-5 pb-4 sm:pb-5 pt-2 border-t border-slate-100">
              <h3 className="text-[10px] sm:text-xs font-semibold text-slate-500 uppercase mb-2">Capabilities</h3>
              <div className="flex flex-wrap gap-1.5">
                {sysInfo.data.capabilities.explainability_methods.map((m) => (
                  <span key={m} className="text-[10px] sm:text-[11px] bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full">
                    {m}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Recent Scans */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100 flex items-center justify-between">
          <h2 className="font-semibold text-sm sm:text-base text-slate-800">Recent Scans</h2>
          <span className="text-[10px] sm:text-xs text-slate-400">{scanHistory.length} in session</span>
        </div>
        {scanHistory.length > 0 ? (
          <div className="divide-y divide-slate-100">
            {scanHistory.slice(0, 5).map((scan) => (
              <div key={scan.id} className="px-4 sm:px-5 py-3 flex items-center gap-3 sm:gap-4">
                <img src={scan.imagePreview} alt="" className="w-10 h-10 rounded-lg object-cover border border-slate-200 shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-xs sm:text-sm font-medium text-slate-700 truncate">
                    {scan.result.total_detected} disease{scan.result.total_detected !== 1 ? "s" : ""} detected
                  </div>
                  <div className="text-[10px] sm:text-xs text-slate-400 truncate">
                    {new Date(scan.timestamp).toLocaleTimeString()} | {scan.result.inference_ms}ms
                  </div>
                </div>
                <span className={`text-[10px] sm:text-xs px-2 sm:px-2.5 py-0.5 sm:py-1 rounded-full font-medium shrink-0 ${
                  scan.result.clinical.referral_priority === "URGENT"
                    ? "bg-red-100 text-red-700"
                    : scan.result.clinical.referral_priority === "ROUTINE"
                      ? "bg-amber-100 text-amber-700"
                      : "bg-emerald-100 text-emerald-700"
                }`}>
                  {scan.result.clinical.referral_priority}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="px-4 sm:px-5 py-8 sm:py-10 text-center">
            <svg className="mx-auto h-10 w-10 text-slate-200 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
            <p className="text-sm text-slate-400">No scans yet</p>
            <button onClick={() => setPage("screening")} className="text-sm text-teal-600 font-medium mt-1 hover:text-teal-700">
              Start a screening
            </button>
          </div>
        )}
      </div>

      {/* Disclaimer */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 sm:p-4 flex items-start gap-2 sm:gap-3">
        <svg className="w-4 sm:w-5 h-4 sm:h-5 text-amber-500 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.27 16.5c-.77.833.192 2.5 1.732 2.5z" />
        </svg>
        <div>
          <p className="text-xs sm:text-sm font-medium text-amber-800">Clinical Decision Support Tool</p>
          <p className="text-[11px] sm:text-xs text-amber-700 mt-0.5">
            This system is designed as a screening aid for qualified ophthalmologists. All AI findings require
            clinical confirmation. Not approved for standalone diagnostic use.
          </p>
        </div>
      </div>
    </div>
  );
}
