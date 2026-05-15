"use client";
import { NavSidebar, MobileTopBar, MobileBottomNav } from "@/components/nav-sidebar";
import { RenewalBanner } from "@/components/billing/renewal-banner";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen-dynamic overflow-hidden bg-slate-50">
      <NavSidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <MobileTopBar />
        <RenewalBanner variant="global" />
        <main id="main-content" className="flex-1 overflow-y-auto pb-20 lg:pb-0">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4 sm:py-6">{children}</div>
        </main>
        <MobileBottomNav />
      </div>
    </div>
  );
}
