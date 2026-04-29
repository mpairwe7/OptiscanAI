"use client";
import { useQuery } from "@tanstack/react-query";
import { fetchKnowledgeGraph } from "@/lib/api";

const CATEGORY_COLORS: Record<string, string> = {
  VASCULAR: "bg-red-50 text-red-700 border-red-100",
  DEGENERATIVE: "bg-amber-50 text-amber-700 border-amber-100",
  GLAUCOMATOUS: "bg-purple-50 text-purple-700 border-purple-100",
  CATARACT: "bg-blue-50 text-blue-700 border-blue-100",
  INFECTIOUS_IMMUNOLOGIC: "bg-orange-50 text-orange-700 border-orange-100",
  HEMATOLOGIC: "bg-pink-50 text-pink-700 border-pink-100",
  NEURO_OPHTHALMIC: "bg-indigo-50 text-indigo-700 border-indigo-100",
  TRACTIONAL: "bg-teal-50 text-teal-700 border-teal-100",
  RETINAL_DETACHMENT_COMPLEX: "bg-rose-50 text-rose-700 border-rose-100",
  STRUCTURAL: "bg-slate-100 text-slate-700 border-slate-200",
};

const SEVERITY_DOT = ["bg-slate-300", "bg-emerald-400", "bg-amber-400", "bg-red-500"];

export function KnowledgeGraphPanel() {
  const { data: kg, isLoading } = useQuery({
    queryKey: ["knowledge-graph"],
    queryFn: fetchKnowledgeGraph,
    staleTime: 60_000 * 30,
  });

  if (isLoading || !kg) return null;

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="px-5 py-4 bg-slate-50 border-b border-slate-100">
        <h3 className="font-semibold text-sm text-slate-700">Clinical Knowledge Graph</h3>
        <p className="text-[10px] text-slate-500 mt-0.5">
          {kg.diseases} diseases | {kg.edges} relationships | Uganda epidemiology
        </p>
      </div>

      <div className="p-5 space-y-5">
        {/* Categories */}
        <div>
          <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide mb-2">Disease Categories</h4>
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(kg.categories).map(([cat, diseases]) => (
              <div key={cat} className={`text-[11px] px-2.5 py-1 rounded-full border ${CATEGORY_COLORS[cat] ?? "bg-slate-50 text-slate-600 border-slate-200"}`}>
                {cat.replace(/_/g, " ")} ({(diseases as string[]).length})
              </div>
            ))}
          </div>
        </div>

        {/* Prevalence */}
        <div>
          <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide mb-2">Uganda Prevalence</h4>
          <div className="space-y-1.5">
            {Object.entries(kg.prevalence)
              .sort(([, a], [, b]) => (b as number) - (a as number))
              .slice(0, 8)
              .map(([code, val]) => (
                <div key={code} className="flex items-center gap-2">
                  <span className="text-xs font-mono w-10 text-slate-600">{code}</span>
                  <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div className="h-full bg-teal-500 rounded-full" style={{ width: `${(val as number) * 100}%` }} />
                  </div>
                  <span className="text-xs text-slate-500 w-10 text-right font-mono">{((val as number) * 100).toFixed(0)}%</span>
                </div>
              ))}
          </div>
        </div>

        {/* Severity */}
        <div>
          <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide mb-2">Severity Levels</h4>
          <div className="flex gap-5">
            {[
              { label: "Severe", level: 3, count: Object.values(kg.severity).filter((v) => v === 3).length },
              { label: "Moderate", level: 2, count: Object.values(kg.severity).filter((v) => v === 2).length },
              { label: "Mild", level: 1, count: Object.values(kg.severity).filter((v) => v === 1).length },
            ].map((s) => (
              <div key={s.level} className="flex items-center gap-1.5">
                <div className={`w-2.5 h-2.5 rounded-full ${SEVERITY_DOT[s.level]}`} />
                <span className="text-xs text-slate-600">{s.label}: <strong>{s.count}</strong></span>
              </div>
            ))}
          </div>
        </div>

        {/* Co-occurrences */}
        <div>
          <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide mb-2">Key Co-occurrences</h4>
          <div className="grid grid-cols-2 gap-1.5">
            {kg.relationships.slice(0, 8).map((r, i) => (
              <div key={i} className="text-xs text-slate-600 flex items-center gap-1.5 bg-slate-50 rounded-lg px-2 py-1">
                <span className="font-mono text-teal-700 font-medium">{r.source}</span>
                <svg className="w-3 h-3 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
                <span className="font-mono text-teal-700 font-medium">{r.target}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
