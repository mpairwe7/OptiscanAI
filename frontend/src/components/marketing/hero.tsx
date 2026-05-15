import Link from "next/link";
import { FundusMockup } from "@/components/marketing/fundus-mockup";

export function Hero() {
  return (
    <section className="relative overflow-hidden">
      {/* Animated gradient mesh background */}
      <div aria-hidden className="absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-b from-white via-teal-50/40 to-white" />
        <div
          className="absolute -top-32 -left-32 w-[480px] h-[480px] rounded-full opacity-40 blur-3xl mesh-float"
          style={{ background: "radial-gradient(circle, #2dd4bf 0%, transparent 70%)" }}
        />
        <div
          className="absolute -top-10 right-0 w-[520px] h-[520px] rounded-full opacity-30 blur-3xl mesh-float-delayed"
          style={{ background: "radial-gradient(circle, #38bdf8 0%, transparent 70%)" }}
        />
        <div
          className="absolute top-40 left-1/2 -translate-x-1/2 w-[680px] h-[480px] rounded-full opacity-25 blur-3xl"
          style={{ background: "radial-gradient(circle, #818cf8 0%, transparent 70%)" }}
        />
        {/* Grid pattern */}
        <div
          className="absolute inset-0 opacity-[0.04]"
          style={{
            backgroundImage:
              "linear-gradient(to right, #0f172a 1px, transparent 1px), linear-gradient(to bottom, #0f172a 1px, transparent 1px)",
            backgroundSize: "48px 48px",
            maskImage: "radial-gradient(circle at center, black 60%, transparent 100%)",
          }}
        />
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-16 sm:py-24 grid lg:grid-cols-2 gap-12 items-center">
        <div>
          <div className="inline-flex items-center gap-1.5 text-xs font-semibold text-teal-700 bg-white/80 backdrop-blur-sm border border-teal-200 px-3 py-1 rounded-full shadow-sm">
            <span className="w-1.5 h-1.5 rounded-full bg-teal-500 animate-pulse-dot" />
            New · Grad-CAM, SHAP, LIME, Integrated Gradients
          </div>
          <h1 className="mt-5 text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-slate-900 text-balance leading-[1.05]">
            Clinical retinal screening,{" "}
            <span className="bg-gradient-to-r from-teal-600 to-cyan-500 bg-clip-text text-transparent">
              explained.
            </span>
          </h1>
          <p className="mt-5 text-lg sm:text-xl text-slate-600 text-pretty max-w-xl leading-relaxed">
            AI-powered multi-disease retinal screening with clinical knowledge-graph reasoning.
            Detect 45 diseases. Visualise <em>why</em> with Grad-CAM, SHAP, LIME, and ELI5.
          </p>

          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              href="/sign-up"
              className="inline-flex items-center gap-2 px-5 py-3 text-sm font-semibold rounded-xl bg-slate-900 hover:bg-slate-800 text-white shadow-lg shadow-slate-900/20 transition-transform hover:-translate-y-px"
            >
              Try free
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Link>
            <Link
              href="/pricing"
              className="inline-flex items-center px-5 py-3 text-sm font-semibold rounded-xl bg-white/80 backdrop-blur-sm border border-slate-300 hover:bg-white text-slate-700 shadow-sm"
            >
              See pricing
            </Link>
            <Link
              href="/#how-it-works"
              className="inline-flex items-center gap-1.5 px-3 py-3 text-sm font-medium text-slate-600 hover:text-slate-900"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
              See how it works
            </Link>
          </div>

          <div className="mt-5 flex flex-wrap gap-x-5 gap-y-1.5 text-xs text-slate-500">
            <span className="flex items-center gap-1.5">
              <svg className="w-3.5 h-3.5 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              Free tier · 10 scans/mo
            </span>
            <span className="flex items-center gap-1.5">
              <svg className="w-3.5 h-3.5 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              No credit card
            </span>
            <span className="flex items-center gap-1.5">
              <svg className="w-3.5 h-3.5 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              Uganda PDP Act ready
            </span>
          </div>
        </div>

        <FundusMockup className="lg:translate-x-4" />
      </div>
    </section>
  );
}
