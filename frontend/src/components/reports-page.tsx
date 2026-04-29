"use client";
import { useQuery } from "@tanstack/react-query";
import { useAppStore } from "@/stores/app-store";
import { fetchAnalytics } from "@/lib/api";

function VolumeBar({ date, scans, maxScans }: { date: string; scans: number; maxScans: number }) {
  const pct = maxScans > 0 ? Math.max((scans / maxScans) * 100, 2) : 2;
  const label = date.slice(5); // MM-DD
  return (
    <div className="flex flex-col items-center gap-1 min-w-[2rem] flex-1">
      <span className="text-[9px] sm:text-[10px] font-mono text-slate-400">{scans}</span>
      <div className="w-full h-16 sm:h-24 bg-slate-100 rounded-t-md relative flex items-end">
        <div
          className="w-full bg-teal-500 rounded-t-md transition-all"
          style={{ height: `${pct}%` }}
        />
      </div>
      <span className="text-[9px] sm:text-[10px] text-slate-400">{label}</span>
    </div>
  );
}

export function ReportsPage() {
  const { scanHistory } = useAppStore();
  const analytics = useQuery({ queryKey: ["analytics"], queryFn: fetchAnalytics, staleTime: 30_000 });

  const urgentCount = scanHistory.filter(
    (s) => s.result.clinical.referral_priority === "URGENT" || s.result.clinical.referral_priority === "EMERGENCY"
  ).length;
  const avgDetected = scanHistory.length > 0
    ? (scanHistory.reduce((a, s) => a + s.result.total_detected, 0) / scanHistory.length).toFixed(1)
    : "0";

  return (
    <div className="space-y-4 sm:space-y-6 animate-fade-in">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-800">Reports & Analytics</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Screening volume, disease trends, and operational metrics</p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        <div className="bg-white rounded-xl border border-slate-200 p-3 sm:p-4">
          <div className="text-[10px] sm:text-xs text-slate-500 mb-1">Session Scans</div>
          <div className="text-xl sm:text-2xl font-bold text-slate-800">{scanHistory.length}</div>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-3 sm:p-4">
          <div className="text-[10px] sm:text-xs text-slate-500 mb-1">Urgent Referrals</div>
          <div className="text-xl sm:text-2xl font-bold text-red-600">{urgentCount}</div>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-3 sm:p-4">
          <div className="text-[10px] sm:text-xs text-slate-500 mb-1">Avg Diseases/Scan</div>
          <div className="text-xl sm:text-2xl font-bold text-slate-800">{avgDetected}</div>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-3 sm:p-4">
          <div className="text-[10px] sm:text-xs text-slate-500 mb-1">Total (All Time)</div>
          <div className="text-xl sm:text-2xl font-bold text-teal-700">{analytics.data?.total_scans ?? 0}</div>
        </div>
      </div>

      {/* Daily Volume Chart */}
      {analytics.data?.daily_volumes && analytics.data.daily_volumes.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100">
            <h2 className="font-semibold text-sm sm:text-base text-slate-800">Daily Scan Volume (Last 14 Days)</h2>
          </div>
          <div className="p-3 sm:p-5 overflow-x-auto">
            <div className="flex gap-1 sm:gap-1.5 min-w-[20rem]">
              {analytics.data.daily_volumes.map((d) => (
                <VolumeBar
                  key={d.date}
                  date={d.date}
                  scans={d.scans}
                  maxScans={Math.max(...analytics.data!.daily_volumes.map((v) => v.scans))}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Top Detected Diseases */}
      {analytics.data?.top_detected_diseases && analytics.data.top_detected_diseases.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100">
            <h2 className="font-semibold text-sm sm:text-base text-slate-800">Most Detected Diseases</h2>
          </div>
          <div className="p-3 sm:p-5 space-y-2">
            {analytics.data.top_detected_diseases.map((d) => {
              const maxCount = analytics.data!.top_detected_diseases[0].count;
              const pct = maxCount > 0 ? (d.count / maxCount) * 100 : 0;
              return (
                <div key={d.code} className="flex items-center gap-2 sm:gap-3">
                  <span className="text-[10px] sm:text-xs font-mono w-10 sm:w-12 text-slate-600 shrink-0">{d.code}</span>
                  <div className="flex-1 h-2.5 sm:h-3 bg-slate-100 rounded-full overflow-hidden">
                    <div className="h-full bg-teal-500 rounded-full" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="text-[10px] sm:text-xs font-mono w-8 sm:w-10 text-right text-slate-500">{d.count}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Session Scan History */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100">
          <h2 className="font-semibold text-sm sm:text-base text-slate-800">Session Scan Log</h2>
        </div>
        {scanHistory.length > 0 ? (
          <>
            {/* Mobile card view */}
            <div className="sm:hidden divide-y divide-slate-100">
              {scanHistory.map((scan) => (
                <div key={scan.id} className="px-4 py-3 space-y-2">
                  <div className="flex items-center gap-3">
                    <img src={scan.imagePreview} alt="" className="w-10 h-10 rounded-lg border object-cover shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-medium text-slate-700">
                        {scan.result.total_detected} disease{scan.result.total_detected !== 1 ? "s" : ""}
                      </div>
                      <div className="text-[10px] text-slate-400">
                        {new Date(scan.timestamp).toLocaleTimeString()} | {scan.result.inference_ms}ms
                      </div>
                    </div>
                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium shrink-0 ${
                      scan.result.clinical.referral_priority === "URGENT" || scan.result.clinical.referral_priority === "EMERGENCY"
                        ? "bg-red-100 text-red-700"
                        : scan.result.clinical.referral_priority === "ROUTINE"
                          ? "bg-amber-100 text-amber-700"
                          : "bg-emerald-100 text-emerald-700"
                    }`}>
                      {scan.result.clinical.referral_priority}
                    </span>
                  </div>
                  <div className="flex gap-1 flex-wrap">
                    {scan.result.predictions.slice(0, 3).map((p) => (
                      <span key={p.code} className="text-[10px] bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded">
                        {p.code}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Desktop table view */}
            <div className="hidden sm:block overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-50 text-left">
                    <th className="px-5 py-2.5 text-xs font-semibold text-slate-500">Time</th>
                    <th className="px-5 py-2.5 text-xs font-semibold text-slate-500">Image</th>
                    <th className="px-5 py-2.5 text-xs font-semibold text-slate-500">Detected</th>
                    <th className="px-5 py-2.5 text-xs font-semibold text-slate-500">Referral</th>
                    <th className="px-5 py-2.5 text-xs font-semibold text-slate-500">Inference</th>
                    <th className="px-5 py-2.5 text-xs font-semibold text-slate-500">Top Diseases</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {scanHistory.map((scan) => (
                    <tr key={scan.id} className="hover:bg-slate-50/50">
                      <td className="px-5 py-2.5 text-xs text-slate-600 font-mono whitespace-nowrap">
                        {new Date(scan.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="px-5 py-2.5">
                        <img src={scan.imagePreview} alt="" className="w-8 h-8 rounded border object-cover" />
                      </td>
                      <td className="px-5 py-2.5">
                        <span className="font-semibold text-slate-700">{scan.result.total_detected}</span>
                      </td>
                      <td className="px-5 py-2.5">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                          scan.result.clinical.referral_priority === "URGENT" || scan.result.clinical.referral_priority === "EMERGENCY"
                            ? "bg-red-100 text-red-700"
                            : scan.result.clinical.referral_priority === "ROUTINE"
                              ? "bg-amber-100 text-amber-700"
                              : "bg-emerald-100 text-emerald-700"
                        }`}>
                          {scan.result.clinical.referral_priority}
                        </span>
                      </td>
                      <td className="px-5 py-2.5 text-xs font-mono text-slate-500">
                        {scan.result.inference_ms}ms
                      </td>
                      <td className="px-5 py-2.5">
                        <div className="flex gap-1">
                          {scan.result.predictions.slice(0, 3).map((p) => (
                            <span key={p.code} className="text-[10px] bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded">
                              {p.code}
                            </span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <div className="px-4 sm:px-5 py-8 sm:py-10 text-center text-slate-400 text-sm">
            No scans recorded in this session yet.
          </div>
        )}
      </div>
    </div>
  );
}
