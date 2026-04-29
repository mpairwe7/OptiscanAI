"use client";
import { NavSidebar, MobileTopBar, MobileBottomNav } from "@/components/nav-sidebar";
import { DashboardPage } from "@/components/dashboard-page";
import { ScreeningPage } from "@/components/screening-page";
import { ReportsPage } from "@/components/reports-page";
import { ReviewPage } from "@/components/review-page";
import { SystemPage } from "@/components/system-page";
import { useAppStore } from "@/stores/app-store";

const PAGES = {
  dashboard: DashboardPage,
  screening: ScreeningPage,
  reports: ReportsPage,
  review: ReviewPage,
  system: SystemPage,
} as const;

export default function Home() {
  const { currentPage } = useAppStore();
  const PageComponent = PAGES[currentPage];

  return (
    <div className="flex h-screen-dynamic overflow-hidden bg-slate-50">
      {/* Desktop sidebar (hidden on mobile via internal lg: classes) */}
      <NavSidebar />

      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile top bar with hamburger */}
        <MobileTopBar />

        {/* Main content - bottom padding for mobile nav */}
        <main id="main-content" className="flex-1 overflow-y-auto pb-20 lg:pb-0">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4 sm:py-6">
            <PageComponent />
          </div>
        </main>

        {/* Mobile bottom navigation */}
        <MobileBottomNav />
      </div>
    </div>
  );
}
