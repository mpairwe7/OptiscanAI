"use client";
import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "@/lib/api";
import { useAppStore, type Page } from "@/stores/app-store";

const NAV_ITEMS: { id: Page; label: string; icon: string }[] = [
  { id: "dashboard", label: "Dashboard", icon: "M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" },
  { id: "screening", label: "Screening", icon: "M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" },
  { id: "reports", label: "Reports", icon: "M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" },
  { id: "review", label: "Review Queue", icon: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" },
  { id: "system", label: "System", icon: "M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z M15 12a3 3 0 11-6 0 3 3 0 016 0z" },
];

/** Mobile top bar with hamburger menu */
export function MobileTopBar() {
  const { currentPage, setMobileMenuOpen } = useAppStore();
  const health = useQuery({ queryKey: ["health"], queryFn: fetchHealth, refetchInterval: 10_000 });
  const isOnline = health.data?.model_loaded;
  const pageLabel = NAV_ITEMS.find((n) => n.id === currentPage)?.label ?? "Dashboard";

  return (
    <header className="lg:hidden sticky top-0 z-30 bg-slate-900 px-4 py-3 flex items-center justify-between safe-bottom">
      <button
        onClick={() => setMobileMenuOpen(true)}
        className="w-10 h-10 flex items-center justify-center rounded-lg text-slate-300 hover:bg-slate-800 active:bg-slate-700"
        aria-label="Open navigation menu"
      >
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
      <div className="flex items-center gap-2">
        <span className="w-7 h-7 rounded-lg bg-teal-500 inline-flex items-center justify-center shrink-0"><svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg></span>
        <span className="text-white font-semibold text-sm">{pageLabel}</span>
      </div>
      <div className="flex items-center gap-2">
        <span
          className={`w-2.5 h-2.5 rounded-full ${
            health.isError ? "bg-red-500" : isOnline ? "bg-emerald-400 animate-pulse-dot" : "bg-amber-400"
          }`}
        />
      </div>
    </header>
  );
}

/** Mobile bottom navigation bar */
export function MobileBottomNav() {
  const { currentPage, setPage } = useAppStore();
  const BOTTOM_ITEMS = NAV_ITEMS.slice(0, 4); // Show first 4 items

  return (
    <nav className="lg:hidden fixed bottom-0 left-0 right-0 z-30 bg-white border-t border-slate-200 safe-bottom">
      <div className="flex items-stretch">
        {BOTTOM_ITEMS.map((item) => {
          const active = currentPage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setPage(item.id)}
              className={`relative flex-1 flex flex-col items-center justify-center py-2 gap-0.5 min-h-[56px] transition-colors ${
                active ? "text-teal-600" : "text-slate-400 active:text-slate-600"
              }`}
              aria-label={item.label}
              aria-current={active ? "page" : undefined}
            >
              {active && <div className="absolute top-0 left-1/2 -translate-x-1/2 w-8 h-0.5 bg-teal-500 rounded-full" />}
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={active ? 2 : 1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d={item.icon} />
              </svg>
              <span className="text-[10px] font-medium leading-tight">{item.label.split(" ")[0]}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}

export function NavSidebar() {
  const { currentPage, setPage, sidebarCollapsed, toggleSidebar, mobileMenuOpen, setMobileMenuOpen } = useAppStore();
  const health = useQuery({ queryKey: ["health"], queryFn: fetchHealth, refetchInterval: 10_000 });

  const isOnline = health.data?.model_loaded;
  const isDemo = health.data && !health.data.model_loaded;

  // Close mobile menu on escape key + body scroll lock
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && mobileMenuOpen) setMobileMenuOpen(false);
    };
    window.addEventListener("keydown", handleKeyDown);
    if (mobileMenuOpen) {
      document.body.classList.add("menu-open");
    } else {
      document.body.classList.remove("menu-open");
    }
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.classList.remove("menu-open");
    };
  }, [mobileMenuOpen, setMobileMenuOpen]);

  return (
    <>
      {/* Mobile overlay */}
      {mobileMenuOpen && (
        <div
          className="mobile-overlay lg:hidden"
          onClick={() => setMobileMenuOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          bg-slate-900 h-full flex flex-col shrink-0 transition-all duration-200
          fixed lg:relative z-50
          ${mobileMenuOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
          w-64 ${sidebarCollapsed ? "lg:w-16" : "lg:w-56"}
        `}
        role="navigation"
        aria-label="Main navigation"
      >
        {/* Brand */}
        <div className="px-4 py-5 flex items-center gap-3 border-b border-slate-700/50">
          <span className="w-8 h-8 rounded-lg bg-teal-500 inline-flex items-center justify-center shrink-0"><svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg></span>
          <div className={`${sidebarCollapsed ? "hidden lg:hidden" : ""} animate-fade-in`}>
            <div className="text-white font-bold text-sm tracking-tight">RetinalAI</div>
            <div className="text-slate-400 text-[10px] font-medium">Clinical Platform v3.0</div>
          </div>
          {/* Mobile close button */}
          <button
            onClick={() => setMobileMenuOpen(false)}
            className="lg:hidden ml-auto w-8 h-8 flex items-center justify-center rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
            aria-label="Close navigation menu"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map((item) => {
            const active = currentPage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setPage(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  active
                    ? "bg-teal-500/15 text-teal-400"
                    : "text-slate-300 hover:bg-slate-800 hover:text-white"
                }`}
                title={sidebarCollapsed ? item.label : undefined}
                aria-label={item.label}
                aria-current={active ? "page" : undefined}
              >
                <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d={item.icon} />
                </svg>
                <span className={`${sidebarCollapsed ? "hidden lg:hidden" : ""} animate-fade-in`}>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Status */}
        <div className="px-3 py-4 border-t border-slate-700/50">
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full shrink-0 ${
                health.isError
                  ? "bg-red-500"
                  : isOnline
                    ? "bg-emerald-400 animate-pulse-dot"
                    : isDemo
                      ? "bg-amber-400 animate-pulse-dot"
                      : "bg-slate-500"
              }`}
            />
            <span className={`text-xs text-slate-300 ${sidebarCollapsed ? "hidden lg:hidden" : ""} animate-fade-in`}>
              {health.isLoading
                ? "Connecting..."
                : health.isError
                  ? "Offline"
                  : isOnline
                    ? "Model Online"
                    : "Demo Mode"}
            </span>
          </div>
          {health.data && (
            <div className={`text-[10px] text-slate-400 mt-1 ${sidebarCollapsed ? "hidden lg:hidden" : ""} animate-fade-in`}>
              {health.data.device} | {health.data.diseases_count} diseases
            </div>
          )}
        </div>

        {/* Collapse toggle (desktop only) */}
        <button
          onClick={toggleSidebar}
          className="hidden lg:block px-3 py-3 border-t border-slate-700/50 text-slate-500 hover:text-slate-300 transition-colors"
          aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <svg
            className={`w-4 h-4 mx-auto transition-transform ${sidebarCollapsed ? "rotate-180" : ""}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
          </svg>
        </button>
      </aside>
    </>
  );
}
