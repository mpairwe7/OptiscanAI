import Link from "next/link";
import { PLANS, formatPrice } from "@/lib/plans";

export function PricingTeaser() {
  return (
    <section id="pricing" className="py-16 sm:py-24 bg-slate-50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-slate-900 text-balance text-center">
          Pricing that scales with your practice
        </h2>
        <p className="mt-3 text-slate-600 text-pretty text-center max-w-2xl mx-auto">
          Start free. Move to Clinician when you outgrow the free tier. Add seats with Practice.
        </p>
        <div className="mt-12 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {PLANS.map((p) => (
            <div
              key={p.id}
              className={`rounded-2xl border bg-white p-6 ${
                p.highlight ? "border-teal-500 ring-2 ring-teal-500/20" : "border-slate-200"
              }`}
            >
              {p.highlight && (
                <div className="text-[10px] uppercase tracking-wider font-bold text-teal-600 mb-2">
                  Most popular
                </div>
              )}
              <div className="font-bold text-slate-900">{p.name}</div>
              <div className="mt-1 text-sm text-slate-500 line-clamp-2">{p.tagline}</div>
              <div className="mt-4 text-2xl font-bold text-slate-900">
                {formatPrice(p, "monthly")}
              </div>
              <div className="mt-1 text-xs text-slate-500">
                {typeof p.scanQuota === "number" ? `${p.scanQuota.toLocaleString()} scans/mo` : "Unlimited scans"}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-8 text-center">
          <Link
            href="/pricing"
            className="inline-flex items-center px-5 py-2.5 text-sm font-semibold rounded-lg bg-slate-900 hover:bg-slate-800 text-white"
          >
            See full pricing →
          </Link>
        </div>
      </div>
    </section>
  );
}
