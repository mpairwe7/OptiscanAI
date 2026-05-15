import Link from "next/link";
import type { ReactNode } from "react";

export interface LegalSection {
  id: string;
  title: string;
  body: ReactNode;
}

interface Props {
  title: string;
  subtitle?: string;
  lastUpdated: string; // ISO date
  effectiveDate?: string; // ISO date
  sections: LegalSection[];
  /** Footer text shown beneath the doc — e.g. how to contact for questions */
  contact?: ReactNode;
}

export function LegalDoc({
  title,
  subtitle,
  lastUpdated,
  effectiveDate,
  sections,
  contact,
}: Props) {
  const formatted = (iso: string) =>
    new Date(iso).toLocaleDateString("en-UG", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-12 sm:py-16">
      <nav aria-label="Breadcrumb" className="text-xs text-slate-500">
        <Link href="/" className="hover:text-slate-700">
          Home
        </Link>
        <span className="mx-1.5">/</span>
        <span className="text-slate-700">{title}</span>
      </nav>

      <header className="mt-6 pb-8 border-b border-slate-200">
        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-slate-900 text-balance">
          {title}
        </h1>
        {subtitle && (
          <p className="mt-3 text-slate-600 text-pretty">{subtitle}</p>
        )}
        <dl className="mt-5 flex flex-wrap gap-x-6 gap-y-2 text-xs">
          <div>
            <dt className="text-slate-500">Last updated</dt>
            <dd className="font-mono font-semibold text-slate-900">{formatted(lastUpdated)}</dd>
          </div>
          {effectiveDate && (
            <div>
              <dt className="text-slate-500">Effective</dt>
              <dd className="font-mono font-semibold text-slate-900">{formatted(effectiveDate)}</dd>
            </div>
          )}
        </dl>
      </header>

      <aside className="mt-6 rounded-xl border border-amber-200 bg-amber-50/70 p-4 text-sm text-amber-900">
        <div className="font-semibold">Draft pending legal review</div>
        <p className="mt-1 text-amber-800">
          This document is a working draft and not a substitute for legal advice. The final binding
          version will be signed off by counsel before general availability of paid plans.
        </p>
      </aside>

      {/* Table of contents */}
      <nav aria-label="Table of contents" className="mt-8">
        <div className="text-[10px] uppercase tracking-wider font-bold text-slate-500">
          On this page
        </div>
        <ol className="mt-2 grid sm:grid-cols-2 gap-x-6 gap-y-1.5 text-sm">
          {sections.map((s, i) => (
            <li key={s.id}>
              <a href={`#${s.id}`} className="text-teal-700 hover:text-teal-900 underline-offset-4 hover:underline">
                {String(i + 1).padStart(2, "0")} · {s.title}
              </a>
            </li>
          ))}
        </ol>
      </nav>

      <div className="mt-10 space-y-10">
        {sections.map((s, i) => (
          <section key={s.id} id={s.id} className="scroll-mt-24">
            <h2 className="text-xl sm:text-2xl font-semibold text-slate-900 tracking-tight flex items-baseline gap-3">
              <span className="text-xs font-mono text-teal-600">
                {String(i + 1).padStart(2, "0")}
              </span>
              {s.title}
              <a
                href={`#${s.id}`}
                aria-label={`Anchor link to ${s.title}`}
                className="text-slate-300 hover:text-teal-600 text-base"
              >
                #
              </a>
            </h2>
            <div className="mt-3 space-y-3 text-slate-700 leading-relaxed text-[15px]">
              {s.body}
            </div>
          </section>
        ))}
      </div>

      {contact && (
        <footer className="mt-14 pt-8 border-t border-slate-200 text-sm text-slate-600">
          {contact}
        </footer>
      )}
    </div>
  );
}
