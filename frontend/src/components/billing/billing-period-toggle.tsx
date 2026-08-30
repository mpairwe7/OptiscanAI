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
    <div className="inline-flex items-center bg-slate-100 rounded-lg p-1 text-sm font-medium" role="group" aria-label="Billing frequency selection">
      <button
        type="button"
        onClick={() => onChange("monthly")}
        aria-pressed={value === "monthly"}
        className={`min-h-[44px] px-4 py-2 rounded-md transition-colors ${
          value === "monthly" ? "bg-white text-slate-900 shadow-sm font-semibold" : "text-slate-700 hover:text-slate-900"
        }`}
      >
        Monthly
      </button>
      <button
        type="button"
        onClick={() => onChange("annual")}
        aria-pressed={value === "annual"}
        className={`min-h-[44px] px-4 py-2 rounded-md transition-colors flex items-center gap-1.5 ${
          value === "annual" ? "bg-white text-slate-900 shadow-sm font-semibold" : "text-slate-700 hover:text-slate-900"
        }`}
      >
        Annual
        <span className="text-[10px] uppercase tracking-wider font-bold text-teal-800 bg-teal-100 px-1.5 py-0.5 rounded">
          Save 17%
        </span>
      </button>
    </div>
  );
}
