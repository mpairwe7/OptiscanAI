"use client";
import { useAppStore } from "@/stores/app-store";

const PRIORITY_STYLES: Record<string, string> = {
  URGENT: "bg-red-100 text-red-800 border-red-200",
  EMERGENCY: "bg-red-200 text-red-900 border-red-300",
  ROUTINE: "bg-amber-100 text-amber-800 border-amber-200",
  FOLLOW_UP: "bg-emerald-100 text-emerald-800 border-emerald-200",
};

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? "bg-red-500" : pct >= 50 ? "bg-amber-500" : "bg-teal-500";
  return (
    <div className="flex items-center gap-1.5 sm:gap-2">
      <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] sm:text-xs font-mono w-10 sm:w-12 text-right text-slate-600">{pct}%</span>
    </div>
  );
}

export function ResultsPanel() {
  const { result } = useAppStore();
  if (!result) return null;

  const { predictions, clinical, inference_ms, total_detected, threshold } = result;
  const fundusGate = result.fundus_gate;
  const oodWarning = result.ood_warning;

  return (
    <div className="space-y-3 sm:space-y-4 animate-fade-in">
      {/* OOD Warning — post-inference detection */}
      {oodWarning?.flagged && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-3 flex items-start gap-2">
          <svg className="w-4 h-4 text-red-500 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.27 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          <div>
            <p className="text-xs font-semibold text-red-800">Out-of-Distribution Warning</p>
            <p className="text-[11px] text-red-700 mt-0.5">{oodWarning.message}</p>
          </div>
        </div>
      )}

      {/* Summary metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3">
        <div className="bg-white rounded-xl border border-slate-200 p-2.5 sm:p-3 text-center">
          <div className="text-xl sm:text-2xl font-bold text-teal-700">{total_detected}</div>
          <div className="text-[9px] sm:text-[10px] text-slate-500 font-medium uppercase tracking-wide">Detected</div>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-2.5 sm:p-3 text-center">
          <div className="text-xl sm:text-2xl font-bold text-blue-700">{inference_ms}<span className="text-[10px] sm:text-xs font-normal text-slate-400">ms</span></div>
          <div className="text-[9px] sm:text-[10px] text-slate-500 font-medium uppercase tracking-wide">Inference</div>
        </div>
        <div className={`rounded-xl border p-2.5 sm:p-3 text-center ${PRIORITY_STYLES[clinical.referral_priority] ?? "bg-slate-100"}`}>
          <div className="text-sm sm:text-lg font-bold truncate">{clinical.referral_priority}</div>
          <div className="text-[9px] sm:text-[10px] font-medium uppercase tracking-wide opacity-80">Priority</div>
        </div>
        {fundusGate && (
          <div className={`rounded-xl border p-2.5 sm:p-3 text-center ${
            fundusGate.confidence >= 0.8 ? "bg-emerald-50 border-emerald-200" :
            fundusGate.confidence >= 0.55 ? "bg-amber-50 border-amber-200" :
            "bg-red-50 border-red-200"
          }`}>
            <div className={`text-xl sm:text-2xl font-bold ${
              fundusGate.confidence >= 0.8 ? "text-emerald-700" :
              fundusGate.confidence >= 0.55 ? "text-amber-700" : "text-red-700"
            }`}>{Math.round(fundusGate.confidence * 100)}%</div>
            <div className="text-[9px] sm:text-[10px] text-slate-500 font-medium uppercase tracking-wide">Fundus Gate</div>
          </div>
        )}
      </div>

      {/* Predictions table */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-3 sm:px-4 py-2.5 sm:py-3 bg-slate-50 border-b border-slate-100 flex items-center justify-between">
          <h3 className="font-semibold text-xs sm:text-sm text-slate-700">Top Predictions</h3>
          <span className="text-[10px] text-slate-400 font-mono">threshold: {threshold}</span>
        </div>
        {predictions.length > 0 ? (
          <div className="divide-y divide-slate-50">
            {predictions.map((p, i) => (
              <div key={p.code} className="px-3 sm:px-4 py-2 sm:py-2.5 flex items-center gap-2 sm:gap-3">
                <span className="w-5 h-5 sm:w-6 sm:h-6 rounded-full bg-teal-50 text-teal-700 text-[10px] sm:text-xs
                               flex items-center justify-center font-bold shrink-0">
                  {i + 1}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-xs sm:text-sm text-slate-700 truncate">{p.name}</div>
                  <div className="text-[9px] sm:text-[10px] text-slate-400 font-mono">{p.code}</div>
                </div>
                <div className="w-20 sm:w-28 shrink-0">
                  <ConfidenceBar value={p.probability} />
                </div>
                <span className={`hidden sm:inline-block text-[10px] px-2 py-0.5 rounded-full font-semibold shrink-0
                  ${p.confidence === "high" ? "bg-red-100 text-red-700" :
                    p.confidence === "medium" ? "bg-amber-100 text-amber-700" :
                    "bg-emerald-100 text-emerald-700"}`}>
                  {p.confidence}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="px-4 py-6 sm:py-8 text-center text-slate-400 text-xs sm:text-sm">
            No diseases detected above threshold
          </div>
        )}
      </div>

      {/* Clinical disclaimer */}
      {predictions.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-2.5 sm:p-3 flex items-start gap-2">
          <svg className="w-3.5 sm:w-4 h-3.5 sm:h-4 text-amber-500 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.27 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          <p className="text-[11px] sm:text-xs text-amber-700">
            AI screening tool. All findings require confirmation by a qualified ophthalmologist.
            Referral priority: <strong>{clinical.referral_priority}</strong>.
          </p>
        </div>
      )}
    </div>
  );
}
