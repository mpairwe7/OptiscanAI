"use client";
import { useQuery } from "@tanstack/react-query";
import { useAppStore } from "@/stores/app-store";
import { explainReasoning, fetchDiseaseInfo, type DiseaseInfo } from "@/lib/api";
import { useState } from "react";

function SeverityBadge({ level }: { level: number }) {
  const cfg = [
    { label: "None", cls: "bg-slate-100 text-slate-500" },
    { label: "Mild", cls: "bg-emerald-100 text-emerald-700" },
    { label: "Moderate", cls: "bg-amber-100 text-amber-700" },
    { label: "Severe", cls: "bg-red-100 text-red-700" },
  ][level] ?? { label: "?", cls: "bg-slate-100" };
  return <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${cfg.cls}`}>{cfg.label}</span>;
}

function DiseaseCard({ code }: { code: string }) {
  const { data: info } = useQuery({
    queryKey: ["disease-info", code],
    queryFn: () => fetchDiseaseInfo(code),
    staleTime: 60_000 * 10,
  });
  if (!info?.info_available) return null;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-3 sm:p-4 space-y-2 sm:space-y-3 animate-fade-in">
      <div className="flex items-center justify-between gap-2">
        <h4 className="font-semibold text-xs sm:text-sm text-slate-700 truncate">
          {info.name} <span className="text-slate-400 font-mono">({info.code})</span>
        </h4>
        <SeverityBadge level={info.severity ?? 0} />
      </div>
      <p className="text-[11px] sm:text-xs text-slate-600 leading-relaxed">{info.description}</p>
      {info.risk_factors && (
        <div>
          <span className="text-[9px] sm:text-[10px] font-semibold text-slate-500 uppercase tracking-wide">Risk Factors</span>
          <div className="flex flex-wrap gap-1 mt-1 sm:mt-1.5">
            {info.risk_factors.map((r) => (
              <span key={r} className="text-[10px] sm:text-[11px] bg-blue-50 text-blue-700 px-1.5 sm:px-2 py-0.5 rounded-full">{r}</span>
            ))}
          </div>
        </div>
      )}
      {info.treatment && (
        <div>
          <span className="text-[9px] sm:text-[10px] font-semibold text-slate-500 uppercase tracking-wide">Treatment</span>
          <ul className="text-[11px] sm:text-xs text-slate-600 mt-1 sm:mt-1.5 space-y-0.5">
            {info.treatment.map((t) => (
              <li key={t} className="flex items-start gap-1.5">
                <span className="w-1 h-1 rounded-full bg-teal-500 mt-1.5 shrink-0" />
                {t}
              </li>
            ))}
          </ul>
        </div>
      )}
      {info.urgency && (
        <div className="text-[11px] sm:text-xs font-medium text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-2.5 sm:px-3 py-1.5 sm:py-2">
          Referral: {info.urgency}
        </div>
      )}
    </div>
  );
}

export function ClinicalReasoning() {
  const { result } = useAppStore();
  const [expanded, setExpanded] = useState(false);

  const predictions = result?.all_probabilities
    ? Object.fromEntries(
        Object.entries(result.all_probabilities).map(([k, v]) => [k, v.probability])
      )
    : {};

  const { data: reasoning, isLoading } = useQuery({
    queryKey: ["reasoning", JSON.stringify(predictions)],
    queryFn: () => explainReasoning(predictions),
    enabled: !!result && Object.keys(predictions).length > 0,
    staleTime: 60_000,
  });

  if (!result) return null;

  const detectedCodes = result.predictions.map((p) => p.code);

  return (
    <div className="space-y-3 sm:space-y-4">
      {/* Loading state */}
      {isLoading && (
        <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-3">
          <div className="skeleton h-4 w-48" />
          <div className="skeleton h-3 w-full" />
          <div className="skeleton h-3 w-3/4" />
        </div>
      )}

      {/* Knowledge Graph Reasoning */}
      {reasoning?.adjustments && reasoning.adjustments.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-3 sm:px-4 py-2.5 sm:py-3 bg-purple-50/50 border-b border-purple-100">
            <h3 className="font-semibold text-xs sm:text-sm text-purple-800">Knowledge Graph Reasoning</h3>
            <p className="text-[9px] sm:text-[10px] text-purple-600 mt-0.5">
              Predictions adjusted based on 144 clinical co-occurrence relationships
            </p>
          </div>
          <div className="divide-y divide-slate-50">
            {reasoning.adjustments.map((adj) => (
              <div key={adj.disease} className="px-3 sm:px-4 py-2 sm:py-2.5 flex items-start gap-2 sm:gap-3">
                <span className="text-[10px] sm:text-xs font-mono text-purple-600 w-8 sm:w-10 pt-0.5 font-semibold shrink-0">{adj.disease}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-1 sm:gap-2 text-[10px] sm:text-xs">
                    <span className="text-slate-500 font-mono">{(adj.original * 100).toFixed(1)}%</span>
                    <svg className="w-3 h-3 text-purple-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                    <span className="text-purple-700 font-semibold font-mono">{(adj.refined * 100).toFixed(1)}%</span>
                    <span className="text-emerald-600 font-medium">(+{(adj.boost * 100).toFixed(1)}%)</span>
                  </div>
                  <p className="text-[10px] sm:text-[11px] text-slate-500 mt-0.5 line-clamp-2">{adj.reason}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Visual Findings */}
      {reasoning?.visual_findings && Object.keys(reasoning.visual_findings).length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-3 sm:p-4">
          <h3 className="text-[9px] sm:text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Expected Visual Findings</h3>
          <div className="flex flex-wrap gap-1 sm:gap-1.5">
            {Object.entries(reasoning.visual_findings).map(([finding, count]) => (
              <span key={finding} className="text-[10px] sm:text-[11px] bg-blue-50 text-blue-700 px-2 sm:px-2.5 py-0.5 sm:py-1 rounded-full border border-blue-100">
                {finding.replace(/_/g, " ")} {Number(count) > 1 ? `(${count})` : ""}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Treatment Recommendations */}
      {reasoning?.treatment_recommendations && Object.keys(reasoning.treatment_recommendations).length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-3 sm:p-4">
          <h3 className="text-[9px] sm:text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Treatment Considerations</h3>
          <div className="space-y-2 sm:space-y-3">
            {Object.entries(reasoning.treatment_recommendations).map(([disease, treatments]) => (
              <div key={disease}>
                <span className="text-[11px] sm:text-xs font-medium text-slate-700">{disease}</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {(treatments as string[]).map((t) => (
                    <span key={t} className="text-[10px] sm:text-[11px] bg-emerald-50 text-emerald-700 px-1.5 sm:px-2 py-0.5 rounded-full">
                      {t.replace(/_/g, " ")}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Disease Detail Cards */}
      {detectedCodes.length > 0 && (
        <div>
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1.5 text-xs text-teal-600 font-medium hover:text-teal-700 active:text-teal-800 transition-colors py-1"
          >
            <svg className={`w-3 h-3 transition-transform ${expanded ? "rotate-90" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
            {expanded ? "Hide" : "Show"} disease details ({detectedCodes.length} detected)
          </button>
          {expanded && (
            <div className="mt-2 sm:mt-3 space-y-2 sm:space-y-3">
              {detectedCodes.map((code) => (
                <DiseaseCard key={code} code={code} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
