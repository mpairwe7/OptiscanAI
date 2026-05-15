import Image from "next/image";
import Link from "next/link";

export function MarketingFooter() {
  return (
    <footer className="border-t border-slate-200 bg-slate-900 text-slate-300">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-14 grid gap-10 md:grid-cols-12 text-sm">
        <div className="md:col-span-5">
          <Link href="/" className="flex items-center gap-2">
            <Image src="/logo.png" alt="OptiscanAI" width={28} height={28} className="w-7 h-7 rounded-lg" />
            <span className="font-bold text-white tracking-tight text-base">OptiscanAI</span>
          </Link>
          <p className="mt-4 text-slate-400 leading-relaxed max-w-sm">
            AI-powered multi-disease retinal screening with explainable AI and clinical knowledge-graph
            reasoning. Built for Ugandan healthcare by MakStartup.
          </p>

          <form
            action="mailto:hello@makstartup.com"
            method="post"
            encType="text/plain"
            className="mt-6 flex gap-2 max-w-md"
          >
            <label className="sr-only" htmlFor="newsletter-email">
              Work email
            </label>
            <input
              id="newsletter-email"
              type="email"
              name="email"
              placeholder="Work email"
              className="flex-1 px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-white placeholder-slate-500 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 text-sm"
            />
            <button
              type="submit"
              className="px-4 py-2 rounded-lg bg-teal-500 hover:bg-teal-400 text-white text-sm font-semibold"
            >
              Subscribe
            </button>
          </form>
          <p className="mt-2 text-xs text-slate-500">
            Product updates and clinical-AI research — once a month, never spam.
          </p>
        </div>

        <div className="md:col-span-2">
          <div className="font-semibold text-white mb-3">Product</div>
          <ul className="space-y-2 text-slate-400">
            <li><Link href="/pricing" className="hover:text-white">Pricing</Link></li>
            <li><Link href="/#how-it-works" className="hover:text-white">How it works</Link></li>
            <li><Link href="/sign-up" className="hover:text-white">Start free</Link></li>
            <li><Link href="/#faq" className="hover:text-white">FAQ</Link></li>
          </ul>
        </div>

        <div className="md:col-span-2">
          <div className="font-semibold text-white mb-3">Company</div>
          <ul className="space-y-2 text-slate-400">
            <li><Link href="/contact-sales" className="hover:text-white">Contact sales</Link></li>
            <li>
              <a href="mailto:hello@makstartup.com" className="hover:text-white">
                hello@makstartup.com
              </a>
            </li>
            <li>
              <a
                href="https://makstartup.com"
                target="_blank"
                rel="noreferrer"
                className="hover:text-white"
              >
                MakStartup ↗
              </a>
            </li>
          </ul>
        </div>

        <div className="md:col-span-3">
          <div className="font-semibold text-white mb-3">Compliance</div>
          <ul className="space-y-2 text-slate-400">
            <li>Uganda PDP Act 2019</li>
            <li>EU AI Act conformity-ready</li>
            <li>EU MDR Class IIa pathway</li>
            <li>FHIR R5 + DICOM interop</li>
          </ul>
        </div>
      </div>

      <div className="border-t border-slate-800">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-5 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500">
          <div>© 2026 OptiscanAI · A MakStartup project</div>
          <div className="flex items-center gap-4">
            <Link href="/legal/privacy" className="hover:text-white">Privacy</Link>
            <Link href="/legal/terms" className="hover:text-white">Terms</Link>
            <span className="font-mono">
              <span className="text-emerald-400">●</span> All systems normal
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
