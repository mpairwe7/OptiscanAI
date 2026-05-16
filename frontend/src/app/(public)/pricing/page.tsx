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
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-3 pr-4 font-semibold text-slate-700">Feature</th>
                {PLANS.map((p) => (
                  <th key={p.id} className="text-left py-3 px-4 font-semibold text-slate-700 whitespace-nowrap">
                    {p.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {FEATURE_MATRIX.map((row) => (
                <tr key={row.key} className="border-b border-slate-100 hover:bg-slate-50/50">
                  <td className="py-3 pr-4 text-slate-700">{row.label}</td>
                  {row.byPlan.map((cell, idx) => (
                    <td key={idx} className="py-3 px-4 text-slate-600 whitespace-nowrap">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-16 text-center text-sm text-slate-500">
        Prices in USD, charged in UGX at the current FX rate. Pay with MTN Mobile Money.
      </div>
    </div>
  );
}
