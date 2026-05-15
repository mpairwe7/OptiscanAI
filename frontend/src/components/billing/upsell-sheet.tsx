"use client";
import Link from "next/link";
import { useBillingStore } from "@/stores/billing-store";
import { planById, type PlanId } from "@/lib/plans";

export function UpsellSheet() {
  const { upsellOpen, upsellPayload, closeUpsell } = useBillingStore();
  if (!upsellOpen || !upsellPayload) return null;

  const required = planById(upsellPayload.required_plan as PlanId);
  if (!required) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 mobile-overlay"
      role="dialog"
      aria-modal="true"
    >
      <div className="relative w-full max-w-md bg-white rounded-2xl shadow-2xl p-6 animate-slide-up">
        <button
          onClick={closeUpsell}
          className="absolute top-3 right-3 w-8 h-8 flex items-center justify-center rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100"
          aria-label="Close"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <div className="w-10 h-10 rounded-full bg-teal-50 text-teal-600 flex items-center justify-center">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
          </svg>
        </div>
        <h2 className="mt-3 text-xl font-bold text-slate-900">
          {required.name} feature
        </h2>
        <p className="mt-1.5 text-sm text-slate-600">{upsellPayload.message}</p>

        <div className="mt-5 rounded-xl border border-teal-200 bg-teal-50/50 p-4">
          <div className="font-semibold text-slate-900">{required.name}</div>
          <div className="mt-0.5 text-xs text-slate-600">{required.description}</div>
          <div className="mt-3 text-2xl font-bold text-slate-900">
            {required.priceUsd.monthly === "contact"
              ? "Contact sales"
              : `$${required.priceUsd.monthly}/mo`}
          </div>
        </div>

        <div className="mt-5 flex flex-col gap-2">
          <Link
            href={
              required.cta.href.startsWith("/app/checkout/")
                ? `${required.cta.href}?cycle=monthly`
                : required.cta.href
            }
            className="inline-flex items-center justify-center px-4 py-2.5 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white"
            onClick={closeUpsell}
          >
            {required.cta.label}
          </Link>
          <Link
            href="/pricing"
            className="inline-flex items-center justify-center px-4 py-2.5 text-sm font-medium text-slate-600 hover:text-slate-900"
            onClick={closeUpsell}
          >
            Compare all plans
          </Link>
        </div>
      </div>
    </div>
  );
}
