"use client";
import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";

export function MarketingNav() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    function onScroll() {
      setScrolled(window.scrollY > 8);
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={`sticky top-0 z-30 transition-all ${
        scrolled
          ? "bg-white/90 backdrop-blur-md border-b border-slate-200 shadow-sm"
          : "bg-white/60 backdrop-blur-sm border-b border-transparent"
      }`}
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between gap-4">
        <Link href="/" className="flex items-center gap-2 shrink-0">
          <Image
            src="/logo.png"
            alt="OptiscanAI"
            width={32}
            height={32}
            className="w-8 h-8 rounded-lg"
            priority
          />
          <span className="font-bold text-slate-900 tracking-tight">OptiscanAI</span>
        </Link>
        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600">
          <Link href="/#how-it-works" className="hover:text-slate-900">How it works</Link>
          <Link href="/pricing" className="hover:text-slate-900">Pricing</Link>
          <Link href="/#faq" className="hover:text-slate-900">FAQ</Link>
          <Link href="/contact-sales" className="hover:text-slate-900">Contact sales</Link>
        </nav>
        <div className="hidden md:flex items-center gap-2">
          <Link
            href="/sign-in"
            className="inline-flex items-center justify-center px-3 py-1.5 text-sm font-medium text-slate-700 hover:text-slate-900"
          >
            Sign in
          </Link>
          <Link
            href="/sign-up"
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-sm font-semibold rounded-lg bg-slate-900 hover:bg-slate-800 text-white"
          >
            Get started
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
        </div>

        {/* Mobile hamburger */}
        <button
          className="md:hidden w-10 h-10 flex items-center justify-center rounded-lg text-slate-700 hover:bg-slate-100"
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            {open ? (
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
            )}
          </svg>
        </button>
      </div>

      {/* Mobile drawer */}
      {open && (
        <div className="md:hidden border-t border-slate-200 bg-white">
          <nav className="px-4 py-3 flex flex-col gap-1 text-sm font-medium">
            <Link href="/#how-it-works" onClick={() => setOpen(false)} className="py-2 px-2 text-slate-700 hover:text-slate-900 rounded-lg hover:bg-slate-50">How it works</Link>
            <Link href="/pricing" onClick={() => setOpen(false)} className="py-2 px-2 text-slate-700 hover:text-slate-900 rounded-lg hover:bg-slate-50">Pricing</Link>
            <Link href="/#faq" onClick={() => setOpen(false)} className="py-2 px-2 text-slate-700 hover:text-slate-900 rounded-lg hover:bg-slate-50">FAQ</Link>
            <Link href="/contact-sales" onClick={() => setOpen(false)} className="py-2 px-2 text-slate-700 hover:text-slate-900 rounded-lg hover:bg-slate-50">Contact sales</Link>
            <div className="mt-2 pt-3 border-t border-slate-100 flex flex-col gap-2">
              <Link
                href="/sign-in"
                onClick={() => setOpen(false)}
                className="px-3 py-2 text-sm font-semibold rounded-lg border border-slate-300 text-slate-700 text-center"
              >
                Sign in
              </Link>
              <Link
                href="/sign-up"
                onClick={() => setOpen(false)}
                className="px-3 py-2 text-sm font-semibold rounded-lg bg-slate-900 text-white text-center"
              >
                Get started
              </Link>
            </div>
          </nav>
        </div>
      )}
    </header>
  );
}
