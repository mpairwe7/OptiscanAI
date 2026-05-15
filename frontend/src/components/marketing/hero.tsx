import Link from "next/link";
import { FundusMockup } from "@/components/marketing/fundus-mockup";

export function Hero() {
  return (
    <section className="hero-section">
      {/* ── Layered background (light healthcare adaptation of ionatec's stack) ── */}
      <div aria-hidden className="hero-bg">
        <div className="hero-grid" />
        <div className="hero-aurora hero-aurora-one" />
        <div className="hero-aurora hero-aurora-two" />
        <div className="hero-stars" />
        <div className="hero-fade-bottom" />
      </div>

      <div className="hero-container">
        {/* Left: copy column (1.04fr) */}
        <div className="hero-copy">
          {/* Live ribbon — amber dot signals clinical attention, not marketing live */}
          <div className="hero-badge">
            <span className="hero-live-dot" aria-hidden />
            <span>
              <span className="font-bold tracking-wide uppercase">Live</span>
              <span className="mx-1.5 text-slate-300">·</span>
              <span>Clinically validated across 6 Ugandan facilities</span>
            </span>
          </div>

          <h1 className="hero-headline">
            Clinical retinal screening,{" "}
            <span className="hero-headline-grad">explained.</span>
          </h1>

          <p className="hero-lede">
            AI-powered multi-disease retinal screening with knowledge-graph clinical reasoning.
            Detect <strong className="text-slate-900">45 diseases</strong> and visualise <em>why</em>{" "}
            with Grad-CAM, SHAP, LIME, Integrated Gradients, and ELI5.
          </p>
          <p className="hero-subdued">
            Built for clinicians who can&apos;t defend a black-box decision in a case review.
          </p>

          {/* Primary CTA row */}
          <div className="hero-cta-row">
            <Link href="/sign-up" className="hero-btn-primary">
              <span>Try free</span>
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5} aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Link>
            <Link href="/pricing" className="hero-btn-secondary">
              See pricing
            </Link>
            <Link href="/#how-it-works" className="hero-btn-tertiary">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24" aria-hidden>
                <path d="M8 5v14l11-7z" />
              </svg>
              <span>Watch the 90-sec walkthrough</span>
            </Link>
          </div>

          {/* Mini check-strip — purchasing signals */}
          <ul className="hero-checks">
            <li>
              <svg className="w-3.5 h-3.5 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3} aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              Free tier · 10 scans/mo
            </li>
            <li>
              <svg className="w-3.5 h-3.5 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3} aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              No credit card required
            </li>
            <li>
              <svg className="w-3.5 h-3.5 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3} aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              Uganda PDP Act 2019 aware
            </li>
          </ul>

          {/* Regulatory chip strip — healthcare trust devices */}
          <div className="hero-reg-row">
            <span className="hero-reg-chip">
              <span className="text-amber-500">●</span> EU AI Act conformity-ready
            </span>
            <span className="hero-reg-chip">
              <span className="text-amber-500">●</span> FHIR R5 reports
            </span>
            <span className="hero-reg-chip">
              <span className="text-amber-500">●</span> Class IIa pathway
            </span>
            <span className="hero-reg-chip">
              <span className="text-amber-500">●</span> 99.5 % uptime target
            </span>
          </div>

          {/* Backed-by line */}
          <p className="hero-backed">
            Built at <strong>Makerere</strong> · Reviewed by ophthalmologists at <strong>Mulago</strong> ·
            Hosted on <strong>Crane Cloud</strong>
          </p>
        </div>

        {/* Right: visual panel (0.96fr) — fundus mockup inside a glass card */}
        <div className="hero-visual">
          <div className="hero-panel">
            <FundusMockup className="hero-fundus" />
          </div>

          {/* Stats trio below the panel */}
          <dl className="hero-stats">
            <div>
              <dt>Diseases</dt>
              <dd>45</dd>
            </div>
            <div>
              <dt>Median inference</dt>
              <dd>85 <span>ms</span></dd>
            </div>
            <div>
              <dt>AUC (production)</dt>
              <dd>0.888</dd>
            </div>
          </dl>
        </div>
      </div>
    </section>
  );
}
