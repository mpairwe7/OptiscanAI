import Link from "next/link";
import { Hero } from "@/components/marketing/hero";
import { LogoCloud } from "@/components/marketing/logo-cloud";
import { HowItWorks } from "@/components/marketing/how-it-works";
import { ValueProps } from "@/components/marketing/value-props";
import { ComplianceBadges } from "@/components/marketing/compliance-badges";
import { Testimonials } from "@/components/marketing/testimonials";
import { PricingTeaser } from "@/components/marketing/pricing-teaser";
import { FAQ } from "@/components/marketing/faq";

export default function MarketingHome() {
  return (
    <>
      {/* JSON-LD: SoftwareApplication schema for Google Knowledge Graph */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "SoftwareApplication",
            name: "OptiscanAI",
            applicationCategory: "MedicalApplication",
            operatingSystem: "Web",
            offers: {
              "@type": "Offer",
              price: "0",
              priceCurrency: "USD",
              description: "Free tier — 10 scans/month",
            },
            aggregateRating: {
              "@type": "AggregateRating",
              ratingValue: "4.8",
              ratingCount: "12",
            },
            url: "https://www.optiscan.makstartup.com",
            publisher: {
              "@type": "Organization",
              name: "MakStartup",
              url: "https://makstartup.com",
            },
          }),
        }}
      />

      <Hero />
      <LogoCloud />
      <HowItWorks />
      <ValueProps />
      <ComplianceBadges />
      <Testimonials />
      <PricingTeaser />
      <FAQ />

      <section className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white py-20 sm:py-24 relative overflow-hidden">
        <div
          aria-hidden
          className="absolute -top-20 left-1/2 -translate-x-1/2 w-[680px] h-[480px] rounded-full opacity-30 blur-3xl"
          style={{ background: "radial-gradient(circle, #2dd4bf 0%, transparent 70%)" }}
        />
        <div className="max-w-3xl mx-auto px-4 text-center relative">
          <h2 className="text-3xl sm:text-5xl font-bold tracking-tight text-balance">
            Start screening in 5 minutes
          </h2>
          <p className="mt-4 text-slate-300 text-lg text-pretty">
            Free tier. No credit card. Upgrade only when you outgrow 10 scans / month.
          </p>
          <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
            <Link
              href="/sign-up"
              className="inline-flex items-center gap-2 px-6 py-3 text-sm font-semibold rounded-xl bg-teal-500 hover:bg-teal-400 text-white shadow-lg shadow-teal-500/20 transition-transform hover:-translate-y-px"
            >
              Get started free
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Link>
            <Link
              href="/contact-sales"
              className="inline-flex items-center px-6 py-3 text-sm font-semibold rounded-xl bg-white/10 hover:bg-white/20 text-white backdrop-blur-sm border border-white/10"
            >
              Talk to sales
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
