"use client";
import { useEffect } from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "@/lib/api";
import { useAppStore } from "@/stores/app-store";
import { useAuthStore } from "@/stores/auth-store";
import { UsageChip } from "@/components/billing/usage-chip";

interface NavItem {
  href: string;
  label: string;
  icon: string;
}

const NAV_ITEMS: NavItem[] = [
  { href: "/app/dashboard", label: "Dashboard", icon: "M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" },
  { href: "/app/screening", label: "Screening", icon: "M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" },
  { href: "/app/reports", label: "Reports", icon: "M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" },
  { href: "/app/review", label: "Review Queue", icon: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" },
  { href: "/app/system", label: "System", icon: "M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z M15 12a3 3 0 11-6 0 3 3 0 016 0z" },
];

const ACCOUNT_ITEMS: NavItem[] = [
  { href: "/app/usage", label: "Usage", icon: "M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" },
  { href: "/app/billing", label: "Billing", icon: "M2.25 8.25h19.5M2.25 9h19.5m-16.5 5.25h6m-6 2.25h3m-3.75 3h15a2.25 2.25 0 002.25-2.25V6.75A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25v10.5A2.25 2.25 0 004.5 19.5z" },
  { href: "/app/team", label: "Team", icon: "M15 19.128a9.38 9.38 0 002.625.372 9.337 9.337 0 004.121-.952 4.125 4.125 0 00-7.533-2.493M15 19.128v-.003c0-1.113-.285-2.16-.786-3.07M15 19.128v.106A12.318 12.318 0 018.624 21c-2.331 0-4.512-.645-6.374-1.766l-.001-.109a6.375 6.375 0 0111.964-3.07M12 6.375a3.375 3.375 0 11-6.75 0 3.375 3.375 0 016.75 0zm8.25 2.25a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z" },
  { href: "/app/account", label: "Account", icon: "M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" },
];

const ADMIN_ITEMS: NavItem[] = [
  { href: "/app/admin/webhooks", label: "Webhooks", icon: "M11.42 15.17L17.25 21A2.652 2.652 0 0021 17.25l-5.877-5.877M11.42 15.17l2.496-3.03c.317-.384.74-.626 1.208-.766M11.42 15.17l-4.655 5.653a2.548 2.548 0 11-3.586-3.586l6.837-5.63m5.108-.233c.55-.164 1.163-.188 1.743-.14a4.5 4.5 0 004.486-6.336l-3.276 3.277a3.004 3.004 0 01-2.25-2.25l3.276-3.276a4.5 4.5 0 00-6.336 4.486c.091 1.076-.071 2.264-.904 2.95l-.102.085m-1.745 1.437L5.909 7.5H4.5L2.25 3.75l1.5-1.5L7.5 4.5v1.409l4.26 4.26m-1.745 1.437l1.745-1.437m6.615 8.206L15.75 15.75M4.867 19.125h.008v.008h-.008v-.008z" },
];

function isActive(pathname: string, href: string): boolean {
  return pathname === href || pathname.startsWith(href + "/");
}

/** Mobile top bar with hamburger menu */
export function MobileTopBar() {
  const pathname = usePathname();
  const { setMobileMenuOpen } = useAppStore();
  const health = useQuery({ queryKey: ["health"], queryFn: fetchHealth, refetchInterval: 10_000 });
  const isOnline = health.data?.model_loaded;
  const current = [...NAV_ITEMS, ...ACCOUNT_ITEMS].find((n) => isActive(pathname, n.href));
  const pageLabel = current?.label ?? "Dashboard";

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
        <Image src="/logo.png" alt="OptiscanAI" width={28} height={28} className="w-7 h-7 rounded-lg shrink-0" priority />
        <span className="text-white font-semibold text-sm">{pageLabel}</span>
      </div>
      <div className="flex items-center gap-2">
        <UsageChip />
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
  const pathname = usePathname();
  const BOTTOM_ITEMS = NAV_ITEMS.slice(0, 4);

  return (
    <nav className="lg:hidden fixed bottom-0 left-0 right-0 z-30 bg-white border-t border-slate-200 safe-bottom">
      <div className="flex items-stretch">
        {BOTTOM_ITEMS.map((item) => {
          const active = isActive(pathname, item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
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
            </Link>
          );
        })}
      </div>
    </nav>
  );
}

export function NavSidebar() {
  const pathname = usePathname();
  const { sidebarCollapsed, toggleSidebar, mobileMenuOpen, setMobileMenuOpen } = useAppStore();
  const isSuperuser = useAuthStore((s) => s.user?.is_superuser ?? false);
  const health = useQuery({ queryKey: ["health"], queryFn: fetchHealth, refetchInterval: 10_000 });

  const isOnline = health.data?.model_loaded;
  const isDemo = health.data && !health.data.model_loaded;

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
      {mobileMenuOpen && (
        <div className="mobile-overlay lg:hidden" onClick={() => setMobileMenuOpen(false)} aria-hidden="true" />
      )}
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
        <div className="px-4 py-5 flex items-center gap-3 border-b border-slate-700/50">
          <Image src="/logo.png" alt="OptiscanAI" width={32} height={32} className="w-8 h-8 rounded-lg shrink-0" priority />
          <div className={`${sidebarCollapsed ? "hidden lg:hidden" : ""} animate-fade-in`}>
            <div className="text-white font-bold text-sm tracking-tight">OptiscanAI</div>
            <div className="text-slate-400 text-[10px] font-medium">Clinical Platform v3.0</div>
          </div>
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

        <nav className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map((item) => (
            <SidebarLink key={item.href} item={item} active={isActive(pathname, item.href)} collapsed={sidebarCollapsed} onNav={() => setMobileMenuOpen(false)} />
          ))}

          <div className={`px-3 pt-4 pb-1 text-[10px] uppercase tracking-wider text-slate-500 font-bold ${sidebarCollapsed ? "hidden lg:hidden" : ""}`}>
            Account
          </div>

          {ACCOUNT_ITEMS.map((item) => (
            <SidebarLink key={item.href} item={item} active={isActive(pathname, item.href)} collapsed={sidebarCollapsed} onNav={() => setMobileMenuOpen(false)} />
          ))}

          {isSuperuser && (
            <>
              <div className={`px-3 pt-4 pb-1 text-[10px] uppercase tracking-wider text-slate-500 font-bold ${sidebarCollapsed ? "hidden lg:hidden" : ""}`}>
                Admin
              </div>
              {ADMIN_ITEMS.map((item) => (
                <SidebarLink key={item.href} item={item} active={isActive(pathname, item.href)} collapsed={sidebarCollapsed} onNav={() => setMobileMenuOpen(false)} />
              ))}
            </>
          )}
        </nav>

        <div className="px-3 py-3 border-t border-slate-700/50 hidden lg:block">
          <UsageChip />
        </div>

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

function SidebarLink({
  item,
  active,
  collapsed,
  onNav,
}: {
  item: NavItem;
  active: boolean;
  collapsed: boolean;
  onNav: () => void;
}) {
  return (
    <Link
      href={item.href}
      onClick={onNav}
      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
        active ? "bg-teal-500/15 text-teal-400" : "text-slate-300 hover:bg-slate-800 hover:text-white"
      }`}
      title={collapsed ? item.label : undefined}
      aria-label={item.label}
      aria-current={active ? "page" : undefined}
    >
      <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d={item.icon} />
      </svg>
      <span className={`${collapsed ? "hidden lg:hidden" : ""} animate-fade-in`}>{item.label}</span>
    </Link>
  );
}
