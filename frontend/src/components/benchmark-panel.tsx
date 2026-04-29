"use client";
import { useQuery } from "@tanstack/react-query";
import { fetchModelHealth } from "@/lib/api";
import { useAppStore } from "@/stores/app-store";

function LatencyChart() {
  const { scanHistory } = useAppStore();
  const recent = scanHistory.slice(0, 20).reverse();

  if (recent.length < 2) return null;

  const maxMs = Math.max(...recent.map((s) => s.result.inference_ms), 1);
  const slaLine = 100; // p99 SLA threshold
  const slaPct = Math.min((slaLine / maxMs) * 100, 100);

  return (
    <div className="mt-3 sm:mt-4 pt-3 sm:pt-4 border-t border-slate-100">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-[9px] sm:text-[10px] font-semibold text-slate-500 uppercase tracking-wide">
          Session Latency History
        </h4>
        <span className="text-[9px] sm:text-[10px] text-slate-400">{recent.length} scans</span>
      </div>
      <div className="relative h-20 sm:h-28 flex items-end gap-[2px] sm:gap-1">
        {/* SLA threshold line */}
        {maxMs > slaLine && (
          <div
            className="absolute left-0 right-0 border-t border-dashed border-red-300 z-10"
            style={{ bottom: `${slaPct}%` }}
          >
            <span className="absolute -top-3 right-0 text-[8px] sm:text-[9px] text-red-400 font-mono">
              SLA {slaLine}ms
            </span>
          </div>
        )}
        {recent.map((scan, i) => {
          const pct = Math.max((scan.result.inference_ms / maxMs) * 100, 3);
          const overSla = scan.result.inference_ms > slaLine;
          return (
            <div
              key={scan.id}
              className="flex-1 min-w-0 group relative"
              title={`${scan.result.inference_ms}ms | ${scan.result.total_detected} diseases`}
            >
              <div
                className={`w-full rounded-t transition-all ${overSla ? "bg-red-400" : "bg-teal-500"} hover:opacity-80`}
                style={{ height: `${pct}%` }}
              />
              {/* Tooltip on hover */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-20">
                <div className="bg-slate-800 text-white text-[9px] px-2 py-1 rounded shadow-lg whitespace-nowrap">
                  {scan.result.inference_ms}ms | {scan.result.total_detected} found
                </div>
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex justify-between mt-1 text-[8px] sm:text-[9px] text-slate-400 font-mono">
        <span>oldest</span>
        <span>latest</span>
      </div>
    </div>
  );
}

export function BenchmarkPanel() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["model-health"],
    queryFn: fetchModelHealth,
    refetchInterval: 15_000,
  });

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-4 sm:p-6 space-y-3">
        <div className="skeleton h-4 w-40" />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="text-center space-y-2">
              <div className="skeleton h-6 w-16 mx-auto" />
              <div className="skeleton h-3 w-12 mx-auto" />
            </div>
          ))}
        </div>
      </div>
    );
  }
  if (isError || !data) return null;

  const uptime_h = (data.uptime_seconds / 3600).toFixed(1);

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="px-4 sm:px-5 py-3 sm:py-4 bg-slate-50 border-b border-slate-100 flex justify-between items-center">
        <h3 className="font-semibold text-xs sm:text-sm text-slate-700">Live Performance Metrics</h3>
        <span className={`text-[10px] sm:text-xs px-2 sm:px-2.5 py-0.5 sm:py-1 rounded-full font-medium ${
          data.sla_compliant ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"
        }`}>
          SLA {data.sla_compliant ? "OK" : "Violated"}
        </span>
      </div>
      <div className="p-3 sm:p-5">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="text-center">
            <div className="text-lg sm:text-xl font-bold text-slate-800">{data.latency_p50_ms.toFixed(1)}<span className="text-[10px] sm:text-xs font-normal text-slate-400 ml-0.5">ms</span></div>
            <div className="text-[9px] sm:text-[10px] text-slate-500 font-medium uppercase tracking-wide mt-1">P50 Latency</div>
          </div>
          <div className="text-center">
            <div className="text-lg sm:text-xl font-bold text-slate-800">{data.latency_p95_ms.toFixed(1)}<span className="text-[10px] sm:text-xs font-normal text-slate-400 ml-0.5">ms</span></div>
            <div className="text-[9px] sm:text-[10px] text-slate-500 font-medium uppercase tracking-wide mt-1">P95 Latency</div>
          </div>
          <div className="text-center">
            <div className="text-lg sm:text-xl font-bold text-slate-800">{data.throughput_rps.toFixed(1)}<span className="text-[10px] sm:text-xs font-normal text-slate-400 ml-0.5">rps</span></div>
            <div className="text-[9px] sm:text-[10px] text-slate-500 font-medium uppercase tracking-wide mt-1">Throughput</div>
          </div>
          <div className="text-center">
            <div className={`text-lg sm:text-xl font-bold ${data.error_rate > 0.01 ? "text-red-600" : "text-emerald-600"}`}>
              {(data.error_rate * 100).toFixed(2)}%
            </div>
            <div className="text-[9px] sm:text-[10px] text-slate-500 font-medium uppercase tracking-wide mt-1">Error Rate</div>
          </div>
        </div>

        {/* Latency History Chart */}
        <LatencyChart />
      </div>
      <div className="px-3 sm:px-5 pb-3 sm:pb-4 flex justify-between text-[10px] text-slate-400">
        <span>Total: {data.total_predictions}</span>
        <span>Uptime: {uptime_h}h</span>
      </div>
    </div>
  );
}
