"use client";
import Link from "next/link";
import { annualSavingsLabel, formatPrice, type Plan, type BillingPeriod } from "@/lib/plans";

export function PricingCard({ plan, period }: { plan: Plan; period: BillingPeriod }) {
  const price = formatPrice(plan, period);
  const savings = period === "annual" ? annualSavingsLabel(plan) : null;
  const isPaid = plan.priceUgx[period] !== 0 && plan.priceUgx[period] !== "contact";
  const href =
    plan.cta.href.startsWith("/app/checkout/") && isPaid
      ? `${plan.cta.href}?cycle=${period}`
      : plan.cta.href;

  return (
    <div
      className={`rounded-2xl border bg-white p-6 flex flex-col h-full ${
        plan.highlight ? "border-teal-500 ring-2 ring-teal-500/20 shadow-md" : "border-slate-200"
      }`}
    >
      {plan.highlight && (
        <div className="text-[10px] uppercase tracking-wider font-bold text-teal-700 mb-2">
          Most popular
        </div>
      )}
      <div className="font-bold text-slate-900 text-lg">{plan.name}</div>
      <p className="mt-1.5 text-sm text-slate-600 line-clamp-2">{plan.tagline}</p>

      <div className="mt-5">
        <div className="text-3xl font-bold text-slate-900">{price}</div>
        {savings && <div className="mt-1 text-xs font-semibold text-teal-700">{savings}</div>}
        {!savings && plan.priceUgx[period] !== "contact" && (
          <div className="mt-1 text-xs text-slate-500">
            {period === "annual" ? `Billed annually` : `Billed monthly`}
          </div>
        )}
      </div>

      <ul className="mt-6 space-y-2 text-sm text-slate-700 flex-1">
        <li className="flex items-start gap-2">
          <span className="text-teal-700 font-bold mt-0.5">✓</span>
          <span>
            {typeof plan.scanQuota === "number"
              ? `${plan.scanQuota.toLocaleString()} scans / month`
              : "Unlimited scans"}
          </span>
        </li>
        <li className="flex items-start gap-2">
          <span className="text-teal-700 font-bold mt-0.5">✓</span>
          <span>
            {typeof plan.seats === "number"
              ? `${plan.seats} ${plan.seats === 1 ? "seat" : "seats"}`
              : "Unlimited seats"}
          </span>
        </li>
        <li className="flex items-start gap-2">
          <span className="text-teal-700 font-bold mt-0.5">✓</span>
          <span>{plan.description}</span>
        </li>
      </ul>

      <Link
        href={href}
        className={`mt-6 inline-flex items-center justify-center w-full min-h-[44px] px-4 py-2.5 text-sm font-semibold rounded-lg transition-colors ${
          plan.cta.variant === "primary"
            ? "bg-teal-700 hover:bg-teal-800 text-white shadow-sm"
            : "border border-slate-300 hover:bg-slate-50 text-slate-800"
        }`}
      >
        {plan.cta.label}
      </Link>
    </div>
  );
}
