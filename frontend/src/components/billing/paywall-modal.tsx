"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useBillingStore } from "@/stores/billing-store";
import { PLANS, planById, type PlanId } from "@/lib/plans";

function fmtCountdown(resets: Date): string {
  const ms = resets.getTime() - Date.now();
  if (ms <= 0) return "now";
  const total = Math.floor(ms / 1000);
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

export function PaywallModal() {
  const { paywallOpen, paywallPayload, closePaywall } = useBillingStore();
  const [, force] = useState(0);

  useEffect(() => {
    if (!paywallOpen) return;
    const t = setInterval(() => force((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [paywallOpen]);

  if (!paywallOpen || !paywallPayload) return null;

  const resets = new Date(paywallPayload.usage.resets_at);
  const recommended = planById(paywallPayload.recommended_plan as PlanId) ?? PLANS[1];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 mobile-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="paywall-title"
    >
      <div className="relative w-full max-w-md bg-white rounded-2xl shadow-2xl p-6 animate-slide-up">
        <button
          onClick={closePaywall}
          className="absolute top-3 right-3 w-8 h-8 flex items-center justify-center rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100"
          aria-label="Close"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <div className="w-10 h-10 rounded-full bg-amber-50 text-amber-600 flex items-center justify-center">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
          </svg>
        </div>
        <h2 id="paywall-title" className="mt-3 text-xl font-bold text-slate-900">
          Monthly scan limit reached
        </h2>
        <p className="mt-1.5 text-sm text-slate-600">{paywallPayload.message}</p>

        <div className="mt-5 rounded-lg bg-slate-50 p-4 text-sm">
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Usage</span>
            <span className="font-mono font-semibold text-slate-900">
              {paywallPayload.usage.used} / {paywallPayload.usage.limit}
            </span>
          </div>
          <div className="mt-3 h-2 bg-slate-200 rounded-full overflow-hidden">
            <div className="h-full bg-red-500" style={{ width: "100%" }} />
          </div>
          <div className="mt-3 flex items-center justify-between">
            <span className="text-slate-500">Resets in</span>
            <span className="font-mono font-semibold text-slate-900">{fmtCountdown(resets)}</span>
          </div>
        </div>

        <div className="mt-5 rounded-lg border border-teal-200 bg-teal-50/50 p-4">
          <div className="text-[10px] uppercase tracking-wider font-bold text-teal-600">Recommended</div>
          <div className="mt-1 font-semibold text-slate-900">
            Upgrade to {recommended.name}
          </div>
          <div className="mt-0.5 text-xs text-slate-600">
            {typeof recommended.scanQuota === "number"
              ? `${recommended.scanQuota.toLocaleString()} scans/mo`
              : "Unlimited scans"}{" "}
            · {recommended.description}
          </div>
        </div>

        <div className="mt-5 flex flex-col gap-2">
          <Link
            href={
              recommended.cta.href.startsWith("/app/checkout/")
                ? `${recommended.cta.href}?cycle=monthly`
                : recommended.cta.href
            }
            className="inline-flex items-center justify-center px-4 py-2.5 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white"
            onClick={closePaywall}
          >
            Upgrade to {recommended.name}
          </Link>
          <button
            onClick={closePaywall}
            className="inline-flex items-center justify-center px-4 py-2.5 text-sm font-medium text-slate-600 hover:text-slate-900"
          >
            Wait until {resets.toLocaleDateString()}
          </button>
        </div>
      </div>
    </div>
  );
}
