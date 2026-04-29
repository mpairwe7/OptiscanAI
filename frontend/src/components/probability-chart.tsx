"use client";
import { useAppStore } from "@/stores/app-store";

export function ProbabilityChart() {
  const { result } = useAppStore();
  if (!result?.all_probabilities) return null;

  const sorted = Object.entries(result.all_probabilities)
    .map(([code, d]) => ({ code, ...d }))
    .sort((a, b) => b.probability - a.probability);

  const threshold = result.threshold ?? 0.5;

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="px-3 sm:px-4 py-2.5 sm:py-3 bg-slate-50 border-b border-slate-100 flex justify-between items-center">
        <h3 className="font-semibold text-xs sm:text-sm text-slate-700">All 45 Disease Probabilities</h3>
        <span className="text-[9px] sm:text-[10px] text-slate-400 font-mono">threshold: {threshold.toFixed(2)}</span>
      </div>
      <div className="max-h-[360px] sm:max-h-[480px] overflow-y-auto divide-y divide-slate-50">
        {sorted.map((d) => {
          const pct = Math.round(d.probability * 100);
          const above = d.probability > threshold;
          const color = pct >= 80 ? "bg-red-500" : pct >= 50 ? "bg-amber-500" : pct > 5 ? "bg-teal-500" : "bg-slate-300";
          return (
            <div key={d.code} className={`px-3 sm:px-4 py-1.5 flex items-center gap-2 sm:gap-3 ${above ? "bg-red-50/40" : ""}`}>
              <span className={`text-[10px] sm:text-xs font-mono w-8 sm:w-10 shrink-0 ${above ? "font-bold text-red-700" : "text-slate-500"}`}>
                {d.code}
              </span>
              <div className="flex-1 h-1.5 sm:h-2 bg-slate-100 rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${color}`} style={{ width: `${Math.max(pct, 1)}%` }} />
              </div>
              <span className={`text-[10px] sm:text-xs w-8 sm:w-10 text-right font-mono shrink-0 ${above ? "font-bold text-red-700" : "text-slate-400"}`}>
                {pct}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
