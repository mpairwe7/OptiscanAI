import { FEATURE_MATRIX, PLANS } from "@/lib/plans";
import { PricingCards } from "./pricing-cards";

export const metadata = {
  title: "Pricing — OptiscanAI",
  description:
    "Free, Clinician, Practice, and Health System plans for AI-powered retinal screening. Monthly or annual. ~17% off annual.",
};

export default function PricingPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-12 sm:py-16">
      <div className="text-center max-w-2xl mx-auto">
        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-slate-900 text-balance">
          Pricing that scales with your practice
        </h1>
        <p className="mt-3 text-slate-600 text-pretty">
          Start free. Move to Clinician when you outgrow the free tier. Add seats with Practice.
          Talk to sales for Health System.
        </p>
      </div>

      <PricingCards />

      {/* Comparison matrix */}
      <div className="mt-16">
        <h2 className="text-2xl font-bold tracking-tight text-slate-900 mb-6">Compare features</h2>
        <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50">
                <th className="text-left py-3.5 px-4 font-semibold text-slate-900">Feature</th>
                {PLANS.map((p) => (
                  <th key={p.id} className="text-left py-3.5 px-4 font-semibold text-slate-900 whitespace-nowrap">
                    {p.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {FEATURE_MATRIX.map((row) => (
                <tr key={row.key} className="border-b border-slate-100 last:border-0 hover:bg-slate-50/70 transition-colors">
                  <td className="py-3 px-4 text-slate-900 font-medium">{row.label}</td>
                  {row.byPlan.map((cell, idx) => (
                    <td key={idx} className="py-3 px-4 text-slate-700 whitespace-nowrap">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-16 flex flex-col items-center gap-2 text-sm text-slate-600">
        <span className="text-xs uppercase tracking-wider font-semibold text-slate-600">
          Pay with
        </span>
        {/*
          MTN Mobile Money mark — geometric reproduction of the brand:
          yellow rounded rectangle (#FFCC00) with bold black "MTN" wordmark.
          Kept inline so it ships with the bundle (no extra request) and
          stays crisp at any DPR.
        */}
        <span
          className="inline-flex items-center gap-2 rounded-full bg-white border border-slate-200 px-3 py-1.5 shadow-sm"
          role="img"
          aria-label="Pay with MTN Mobile Money"
        >
          <svg
            viewBox="0 0 64 32"
            className="h-6 w-12"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden="true"
            focusable="false"
          >
            <rect x="0" y="0" width="64" height="32" rx="6" fill="#FFCC00" />
            <text
              x="32"
              y="22"
              textAnchor="middle"
              fontFamily="Inter, system-ui, sans-serif"
              fontWeight="900"
              fontSize="18"
              fill="#000000"
              letterSpacing="-0.5"
            >
              MTN
            </text>
          </svg>
          <span className="text-sm font-semibold text-slate-800">MoMo</span>
        </span>
      </div>
    </div>
  );
}
