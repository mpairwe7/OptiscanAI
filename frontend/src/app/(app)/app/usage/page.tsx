"use client";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { apiGetUsage } from "@/lib/auth-api";

export default function UsagePage() {
  const usage = useQuery({ queryKey: ["billing", "usage"], queryFn: apiGetUsage, refetchInterval: 30_000 });
  if (usage.isLoading) return <div className="skeleton h-48 rounded-2xl" />;
  if (!usage.data) return <p>No usage data.</p>;

  const { scan_limit, scans_used, scans_remaining, seat_limit, seats_used, breakdown, period_start, period_end } = usage.data;
  const pct = scan_limit && scan_limit > 0 ? Math.min(100, (scans_used / scan_limit) * 100) : 0;
  const barColor = pct >= 100 ? "bg-red-500" : pct >= 80 ? "bg-amber-500" : "bg-teal-500";

  const labels: Record<string, string> = {
    scan: "Scans",
    explain_gradcam: "Grad-CAM",
    explain_lime: "LIME",
    explain_shap: "SHAP",
    explain_ig: "Integrated Gradients",
    explain_eli5: "ELI5",
    clinical_reasoning: "Clinical reasoning",
    audit_export: "Audit exports",
  };

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900">Usage</h1>
        <p className="mt-1 text-sm text-slate-600">
          Current billing period: {new Date(period_start).toLocaleDateString()} →{" "}
          {new Date(period_end).toLocaleDateString()}
        </p>
      </header>

      <section className="rounded-2xl border border-slate-200 bg-white p-6">
        <div className="flex items-baseline justify-between">
          <h2 className="font-semibold text-slate-900">Scans this period</h2>
          <div className="font-mono text-2xl font-bold text-slate-900">
            {scans_used}
            <span className="text-slate-400 text-base font-normal">
              {" "}
              / {scan_limit === null ? "∞" : scan_limit}
            </span>
          </div>
        </div>
        {scan_limit !== null && (
          <>
            <div className="mt-3 h-2 bg-slate-200 rounded-full overflow-hidden">
              <div className={`h-full ${barColor}`} style={{ width: `${pct}%` }} />
            </div>
            <div className="mt-2 text-xs text-slate-500">
              {scans_remaining} scans remaining
              {pct >= 80 && (
                <>
                  {" · "}
                  <Link href="/pricing" className="text-teal-600 hover:text-teal-700 font-medium">
                    Upgrade for more
                  </Link>
                </>
              )}
            </div>
          </>
        )}
      </section>

      {seat_limit !== null && (
        <section className="rounded-2xl border border-slate-200 bg-white p-6">
          <div className="flex items-baseline justify-between">
            <h2 className="font-semibold text-slate-900">Seats</h2>
            <div className="font-mono text-2xl font-bold text-slate-900">
              {seats_used}
              <span className="text-slate-400 text-base font-normal"> / {seat_limit}</span>
            </div>
          </div>
        </section>
      )}

      <section className="rounded-2xl border border-slate-200 bg-white p-6">
        <h2 className="font-semibold text-slate-900">Breakdown by activity</h2>
        {Object.keys(breakdown).length === 0 ? (
          <p className="mt-3 text-sm text-slate-500">No activity yet this period.</p>
        ) : (
          <table className="mt-4 w-full text-sm">
            <tbody>
              {Object.entries(breakdown).map(([k, v]) => (
                <tr key={k} className="border-b border-slate-100">
                  <td className="py-2 text-slate-700">{labels[k] ?? k}</td>
                  <td className="py-2 text-right font-mono font-semibold text-slate-900">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
