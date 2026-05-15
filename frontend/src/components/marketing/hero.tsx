import Link from "next/link";
import { FundusMockup } from "@/components/marketing/fundus-mockup";

export function Hero() {
  return (
    <section className="relative isolate overflow-hidden bg-white">
      <div aria-hidden className="absolute inset-0 -z-10">
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse 60% 50% at 50% 0%, rgba(20,184,166,0.10), transparent 65%), radial-gradient(ellipse 50% 40% at 90% 10%, rgba(56,189,248,0.08), transparent 70%), linear-gradient(180deg, #ffffff 0%, #fafbfd 50%, #ffffff 100%)",
          }}
        />
        <div
          className="absolute inset-0 opacity-40"
          style={{
            backgroundImage:
              "linear-gradient(rgba(15,118,110,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(15,118,110,0.06) 1px, transparent 1px)",
            backgroundSize: "56px 56px",
            WebkitMaskImage:
              "radial-gradient(ellipse 80% 60% at 50% 30%, #000 25%, transparent 80%)",
            maskImage:
              "radial-gradient(ellipse 80% 60% at 50% 30%, #000 25%, transparent 80%)",
          }}
        />
        <div className="absolute inset-x-0 bottom-0 h-24 bg-gradient-to-b from-transparent to-white" />
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16 sm:py-20 lg:py-24">
        <div className="grid lg:grid-cols-[1.1fr_1fr] gap-12 lg:gap-16 items-center">
          <div className="max-w-xl">
            <div className="inline-flex items-center gap-2 rounded-full border border-teal-200/70 bg-white/80 backdrop-blur px-3 py-1.5 text-xs font-medium text-slate-700 shadow-sm">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-75" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-amber-500" />
              </span>
              <span className="font-bold uppercase tracking-wider text-slate-900">Live</span>
              <span className="text-slate-300">·</span>
              <span>Clinically validated across 6 Uganda facilities</span>
            </div>

            <h1 className="mt-6 text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight leading-[1.05] text-slate-900 text-balance">
              Clinical retinal screening,{" "}
              <span
                className="bg-clip-text text-transparent"
                style={{
                  backgroundImage:
                    "linear-gradient(135deg, #0d9488 0%, #06b6d4 100%)",
                }}
              >
                explained.
              </span>
            </h1>

            <p className="mt-5 text-lg leading-relaxed text-slate-600 text-pretty">
              AI-powered multi-disease retinal screening with knowledge-graph clinical reasoning.
              Detect <strong className="font-semibold text-slate-900">45 diseases</strong> and
              visualise <em>why</em> with Grad-CAM, SHAP, LIME, and Integrated Gradients.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                href="/sign-up"
                className="group inline-flex items-center gap-1.5 rounded-full bg-slate-900 hover:bg-slate-800 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-slate-900/20 transition-all hover:-translate-y-px"
              >
                Try free
                <svg
                  className="w-4 h-4 transition-transform group-hover:translate-x-0.5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2.5}
                  aria-hidden
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </Link>
              <Link
                href="/pricing"
                className="inline-flex items-center rounded-full border border-slate-300 bg-white px-5 py-3 text-sm font-semibold text-slate-900 hover:border-slate-400 hover:bg-slate-50"
              >
                See pricing
              </Link>
            </div>

            <ul className="mt-6 flex flex-wrap gap-x-5 gap-y-2 text-xs text-slate-500">
              <li className="inline-flex items-center gap-1.5">
                <svg className="w-3.5 h-3.5 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3} aria-hidden>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
                Free tier · 10 scans/mo
              </li>
              <li className="inline-flex items-center gap-1.5">
                <svg className="w-3.5 h-3.5 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3} aria-hidden>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
                No credit card
              </li>
              <li className="inline-flex items-center gap-1.5">
                <svg className="w-3.5 h-3.5 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3} aria-hidden>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
                EU AI Act ready
              </li>
            </ul>
          </div>

          <div className="relative">
            <div
              className="relative rounded-3xl border border-slate-200/80 bg-white/70 backdrop-blur-xl p-3 sm:p-4 shadow-[0_30px_70px_-25px_rgba(15,23,42,0.25)]"
            >
              <div
                aria-hidden
                className="absolute -top-px left-[10%] right-[10%] h-px"
                style={{
                  background:
                    "linear-gradient(90deg, transparent, rgba(15,118,110,0.4), transparent)",
                }}
              />
              <FundusMockup />

              <div className="mt-3 grid grid-cols-3 gap-2 rounded-2xl bg-white/80 backdrop-blur border border-slate-200/70 p-3">
                <div className="text-center">
                  <div className="text-[10px] uppercase tracking-wider font-semibold text-slate-400">Diseases</div>
                  <div className="mt-0.5 font-mono text-lg font-bold text-slate-900">45</div>
                </div>
                <div className="text-center border-x border-slate-200/70">
                  <div className="text-[10px] uppercase tracking-wider font-semibold text-slate-400">Inference</div>
                  <div className="mt-0.5 font-mono text-lg font-bold text-slate-900">
                    85<span className="text-xs text-slate-400 ml-0.5">ms</span>
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-[10px] uppercase tracking-wider font-semibold text-slate-400">AUC</div>
                  <div className="mt-0.5 font-mono text-lg font-bold text-slate-900">0.888</div>
                </div>
              </div>
            </div>

            <div
              aria-hidden
              className="hidden lg:block absolute -z-10 inset-0 rounded-3xl translate-x-3 translate-y-3 bg-gradient-to-br from-teal-500/10 via-transparent to-cyan-500/10"
            />
          </div>
        </div>
      </div>
    </section>
  );
}
