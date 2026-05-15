"use client";
import type { BillingPeriod } from "@/lib/plans";

export function BillingPeriodToggle({
  value,
  onChange,
}: {
  value: BillingPeriod;
  onChange: (v: BillingPeriod) => void;
}) {
  return (
    <div className="inline-flex items-center bg-slate-100 rounded-lg p-1 text-sm font-medium">
      <button
        onClick={() => onChange("monthly")}
        className={`px-3.5 py-1.5 rounded-md transition-colors ${
          value === "monthly" ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
        }`}
      >
        Monthly
      </button>
      <button
        onClick={() => onChange("annual")}
        className={`px-3.5 py-1.5 rounded-md transition-colors flex items-center gap-1.5 ${
          value === "annual" ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
        }`}
      >
        Annual
        <span className="text-[10px] uppercase tracking-wider font-bold text-teal-600 bg-teal-50 px-1.5 py-0.5 rounded">
          Save 17%
        </span>
      </button>
    </div>
  );
}
