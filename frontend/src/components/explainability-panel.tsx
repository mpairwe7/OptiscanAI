"use client";
import { useAppStore } from "@/stores/app-store";
import { useAvailableMethods } from "@/hooks/use-explain";

const METHODS = [
  { id: "gradcam", label: "Grad-CAM" },
  { id: "lime", label: "LIME" },
  { id: "shap", label: "SHAP" },
  { id: "ig", label: "IG" },
  { id: "eli5", label: "ELI5" },
] as const;

function GradCAMView() {
  const { gradcamResult } = useAppStore();
  if (!gradcamResult) return null;

  return (
    <div className="space-y-3 animate-fade-in">
      <div className="flex items-center justify-between text-[10px] sm:text-xs text-slate-500">
        <span>Gradient-weighted Class Activation Mapping</span>
        <span className="font-mono">{gradcamResult.elapsed_ms}ms</span>
      </div>
      <div className="grid grid-cols-2 gap-2 sm:gap-3">
        <div>
          <div className="text-[9px] sm:text-[10px] text-slate-500 mb-1 font-medium uppercase tracking-wide">Original</div>
          <img src={gradcamResult.original} alt="Original retinal image" className="w-full rounded-lg border border-slate-200" />
        </div>
        {gradcamResult.heatmaps.map((h) => (
          <div key={h.class_index}>
            <div className="text-[9px] sm:text-[10px] text-slate-500 mb-1 truncate font-medium">
              {h.disease_name} <span className="font-mono text-teal-600">({(h.probability * 100).toFixed(1)}%)</span>
            </div>
            {h.heatmap ? (
              <img src={h.heatmap} alt={`GradCAM heatmap for ${h.disease_name}`} className="w-full rounded-lg border border-slate-200" />
            ) : (
              <div className="w-full aspect-square bg-slate-50 rounded-lg flex items-center justify-center text-[10px] sm:text-xs text-slate-400 border border-slate-200">
                {h.error ?? "No heatmap"}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function LIMEView() {
  const { limeResult } = useAppStore();
  if (!limeResult) return null;

  return (
    <div className="space-y-3 animate-fade-in">
      <div className="flex items-center justify-between text-[10px] sm:text-xs text-slate-500">
        <span>Local Interpretable Model-agnostic Explanations</span>
        <span className="font-mono">{limeResult.elapsed_ms}ms</span>
      </div>
      {Object.entries(limeResult.explanations).map(([name, data]) => (
        <div key={name} className="bg-slate-50 border border-slate-200 rounded-xl p-3 sm:p-4">
          <div className="flex justify-between items-center mb-2 sm:mb-3">
            <span className="font-medium text-xs sm:text-sm text-slate-700 truncate">{name}</span>
            <span className="text-[10px] sm:text-xs font-mono text-slate-500">{((data.prediction ?? 0) * 100).toFixed(1)}%</span>
          </div>
          {"error" in data && data.error ? (
            <div className="text-[10px] sm:text-xs text-red-500">{data.error}</div>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-2 text-xs mb-2 sm:mb-3">
                <div className="bg-emerald-50 border border-emerald-100 rounded-lg p-2 text-center">
                  <div className="font-bold text-emerald-700">{data.summary?.top_positive_features ?? 0}</div>
                  <div className="text-emerald-600 text-[9px] sm:text-[10px]">Supporting</div>
                </div>
                <div className="bg-red-50 border border-red-100 rounded-lg p-2 text-center">
                  <div className="font-bold text-red-700">{data.summary?.top_negative_features ?? 0}</div>
                  <div className="text-red-600 text-[9px] sm:text-[10px]">Opposing</div>
                </div>
              </div>
              {data.feature_weights && (
                <div className="space-y-1.5">
                  {Object.entries(data.feature_weights)
                    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                    .slice(0, 6)
                    .map(([seg, weight]) => {
                      const pct = Math.min(Math.abs(weight) * 500, 100);
                      const positive = weight > 0;
                      return (
                        <div key={seg} className="flex items-center gap-1.5 sm:gap-2 text-[10px] sm:text-xs">
                          <span className="w-12 sm:w-14 text-slate-500 truncate font-mono">Seg {seg}</span>
                          <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                            <div className={`h-full rounded-full ${positive ? "bg-emerald-500" : "bg-red-500"}`}
                                 style={{ width: `${pct}%` }} />
                          </div>
                          <span className={`w-12 sm:w-14 text-right font-mono ${positive ? "text-emerald-600" : "text-red-600"}`}>
                            {weight > 0 ? "+" : ""}{weight.toFixed(3)}
                          </span>
                        </div>
                      );
                    })}
                </div>
              )}
            </>
          )}
        </div>
      ))}
    </div>
  );
}

function SHAPView() {
  const { shapResult } = useAppStore();
  if (!shapResult) return null;

  return (
    <div className="space-y-3 animate-fade-in">
      <div className="flex items-center justify-between text-[10px] sm:text-xs text-slate-500">
        <span>SHapley Additive exPlanations</span>
        <span className="font-mono">{shapResult.elapsed_ms}ms</span>
      </div>
      {Object.entries(shapResult.explanations).map(([name, data]) => (
        <div key={name} className="bg-slate-50 border border-slate-200 rounded-xl p-3 sm:p-4">
          <div className="flex justify-between items-center mb-2 sm:mb-3">
            <span className="font-medium text-xs sm:text-sm text-slate-700 truncate">{name}</span>
            <span className="text-[10px] sm:text-xs font-mono text-slate-500">{((data.prediction ?? 0) * 100).toFixed(1)}%</span>
          </div>
          {"error" in data && data.error ? (
            <div className="text-[10px] sm:text-xs text-red-500">{data.error}</div>
          ) : data.feature_importance ? (
            <div className="space-y-1.5 sm:space-y-2">
              {Object.entries(data.feature_importance).map(([metric, value]) => {
                const maxVal = data.feature_importance.max_abs_shap || 1;
                const pct = Math.min((Math.abs(value) / maxVal) * 100, 100);
                return (
                  <div key={metric} className="flex items-center gap-1.5 sm:gap-2 text-[10px] sm:text-xs">
                    <span className="w-20 sm:w-28 text-slate-600 truncate">{metric.replace(/_/g, " ")}</span>
                    <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-500 rounded-full" style={{ width: `${pct}%` }} />
                    </div>
                    <span className="w-12 sm:w-16 text-right font-mono text-slate-600">{value.toFixed(4)}</span>
                  </div>
                );
              })}
            </div>
          ) : null}
        </div>
      ))}
    </div>
  );
}

function IGView() {
  const { igResult } = useAppStore();
  if (!igResult) return null;

  return (
    <div className="space-y-3 animate-fade-in">
      <div className="flex items-center justify-between text-[10px] sm:text-xs text-slate-500">
        <span>Integrated Gradients Attribution</span>
        <span className="font-mono">{igResult.elapsed_ms}ms</span>
      </div>
      {Object.entries(igResult.explanations).map(([name, data]) => {
        const s = data.attribution_summary;
        return (
          <div key={name} className="bg-slate-50 border border-slate-200 rounded-xl p-3 sm:p-4">
            <div className="flex justify-between items-center mb-2 sm:mb-3">
              <span className="font-medium text-xs sm:text-sm text-slate-700 truncate">{name}</span>
              <span className="text-[10px] sm:text-xs font-mono text-slate-500">{((data.prediction ?? 0) * 100).toFixed(1)}%</span>
            </div>
            {s && (
              <div className="grid grid-cols-3 gap-1.5 sm:gap-2 text-xs">
                <div className="bg-orange-50 border border-orange-100 rounded-lg p-1.5 sm:p-2 text-center">
                  <div className="font-bold text-orange-700 font-mono text-[10px] sm:text-xs">{s.mean.toFixed(4)}</div>
                  <div className="text-orange-600 text-[9px] sm:text-[10px]">Mean</div>
                </div>
                <div className="bg-orange-50 border border-orange-100 rounded-lg p-1.5 sm:p-2 text-center">
                  <div className="font-bold text-orange-700 font-mono text-[10px] sm:text-xs">{s.max.toFixed(4)}</div>
                  <div className="text-orange-600 text-[9px] sm:text-[10px]">Max</div>
                </div>
                <div className="bg-orange-50 border border-orange-100 rounded-lg p-1.5 sm:p-2 text-center">
                  <div className="font-bold text-orange-700 font-mono text-[10px] sm:text-xs">{s.min.toFixed(6)}</div>
                  <div className="text-orange-600 text-[9px] sm:text-[10px]">Min</div>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ELI5View() {
  const { eli5Result } = useAppStore();
  if (!eli5Result) return null;

  return (
    <div className="space-y-3 animate-fade-in">
      <div className="flex items-center justify-between text-[10px] sm:text-xs text-slate-500">
        <span>Human-Readable Explanations</span>
        <span className="font-mono">{eli5Result.elapsed_ms}ms</span>
      </div>
      {Object.entries(eli5Result.explanations).map(([name, data]) => (
        <div key={name} className="bg-slate-50 border border-slate-200 rounded-xl p-3 sm:p-4">
          <div className="flex justify-between items-center mb-2 sm:mb-3">
            <span className="font-medium text-xs sm:text-sm text-slate-700 truncate">{name}</span>
            <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold
              ${data.confidence_level === "High" ? "bg-red-100 text-red-700" :
                data.confidence_level === "Medium" ? "bg-amber-100 text-amber-700" :
                "bg-emerald-100 text-emerald-700"}`}>
              {data.confidence_level}
            </span>
          </div>
          {"error" in data && data.error ? (
            <div className="text-[10px] sm:text-xs text-red-500">{data.error}</div>
          ) : (
            <>
              {data.explanation_text && (
                <div className="text-[11px] sm:text-xs text-slate-600 bg-white border border-slate-100 rounded-lg p-2.5 sm:p-3 mb-2 sm:mb-3 whitespace-pre-line leading-relaxed">
                  {data.explanation_text}
                </div>
              )}
              {data.top_contributing_features && (
                <div className="space-y-1.5">
                  <div className="text-[9px] sm:text-[10px] font-semibold text-slate-500 uppercase tracking-wide mb-1">Top Features</div>
                  {data.top_contributing_features.map((f) => (
                    <div key={f.feature} className="flex items-center gap-1.5 sm:gap-2 text-[10px] sm:text-xs">
                      <span className={`w-2 sm:w-2.5 h-2 sm:h-2.5 rounded-full shrink-0 ${f.direction === "positive" ? "bg-emerald-400" : "bg-red-400"}`} />
                      <span className="flex-1 text-slate-600 truncate">{f.feature.replace(/_/g, " ")}</span>
                      <span className={`font-mono ${f.direction === "positive" ? "text-emerald-600" : "text-red-600"}`}>
                        {f.weight > 0 ? "+" : ""}{f.weight.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      ))}
    </div>
  );
}

export function ExplainabilityPanel() {
  const { result, activeXaiMethod, setActiveXaiMethod, gradcamResult, limeResult, shapResult, igResult, eli5Result } = useAppStore();
  const { data: methods } = useAvailableMethods();

  if (!result) return null;

  const methodAvailability: Record<string, boolean> = {
    gradcam: methods?.methods?.gradcam?.available ?? true,
    lime: methods?.methods?.lime?.available ?? true,
    shap: methods?.methods?.shap?.available ?? false,
    ig: methods?.methods?.integrated_gradients?.available ?? true,
    eli5: methods?.methods?.eli5?.available ?? false,
  };

  const resultMap: Record<string, unknown> = {
    gradcam: gradcamResult, lime: limeResult, shap: shapResult, ig: igResult, eli5: eli5Result,
  };

  const hasCurrentResult = !!resultMap[activeXaiMethod];
  const anyLoading = !gradcamResult && methodAvailability.gradcam;

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      {/* Header — tabs only, no button */}
      <div className="px-3 sm:px-4 py-2.5 sm:py-3 bg-slate-50 border-b border-slate-100 space-y-2 sm:space-y-3">
        <h3 className="font-semibold text-xs sm:text-sm text-slate-700">Model Explainability</h3>
        <div className="flex gap-1.5 overflow-x-auto pb-0.5 -mx-1 px-1">
          {METHODS.map((m) => {
            const active = activeXaiMethod === m.id;
            const available = methodAvailability[m.id];
            const done = !!resultMap[m.id];
            const loading = available && !done;
            return (
              <button
                key={m.id}
                onClick={() => setActiveXaiMethod(m.id)}
                disabled={!available}
                className={`px-2.5 sm:px-3 py-1.5 text-[11px] sm:text-xs font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 flex items-center gap-1
                  ${active
                    ? "bg-teal-600 text-white shadow-sm"
                    : available
                      ? "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 active:bg-slate-100"
                      : "bg-slate-50 text-slate-300 cursor-not-allowed"
                  }`}
                aria-label={`${m.label} explainability method`}
                aria-current={active ? "true" : undefined}
              >
                {loading && <div className="w-2.5 h-2.5 border-[1.5px] border-current/30 border-t-current rounded-full animate-spin" />}
                {done && <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>}
                {m.label}
                {!available && " (N/A)"}
              </button>
            );
          })}
        </div>
      </div>

      {/* Content */}
      <div className="p-3 sm:p-4">
        {!hasCurrentResult && (
          <div className="text-center py-8 sm:py-10">
            <div className="inline-block h-6 w-6 animate-spin rounded-full border-2 border-teal-600 border-t-transparent" />
            <p className="mt-2 text-xs sm:text-sm text-slate-500">
              Analysing with {METHODS.find((m) => m.id === activeXaiMethod)?.label}...
            </p>
          </div>
        )}

        {activeXaiMethod === "gradcam" && <GradCAMView />}
        {activeXaiMethod === "lime" && <LIMEView />}
        {activeXaiMethod === "shap" && <SHAPView />}
        {activeXaiMethod === "ig" && <IGView />}
        {activeXaiMethod === "eli5" && <ELI5View />}
      </div>
    </div>
  );
}
