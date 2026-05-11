"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "@/lib/api";
import { useAppStore } from "@/stores/app-store";

export function SettingsSidebar() {
  const { threshold, setThreshold, topK, setTopK } = useAppStore();
  const health = useQuery({ queryKey: ["health"], queryFn: fetchHealth, refetchInterval: 10_000 });
  const [open, setOpen] = useState(false);

  const content = (
    <div className="space-y-6">
      <div>
        <h2 className="font-bold text-lg text-teal-700">OptiscanAI</h2>
        <p className="text-xs text-gray-500 mt-0.5">Multi-disease screening v3.0</p>
      </div>

      {/* API Status */}
      <div className="rounded-lg border p-3 space-y-2">
        <h3 className="text-xs font-semibold text-gray-500 uppercase">System Status</h3>
        {health.isLoading ? (
          <div className="space-y-2">
            <div className="skeleton h-4 w-24" />
            <div className="skeleton h-3 w-32" />
          </div>
        ) : health.isError ? (
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-red-500" />
            <span className="text-xs text-red-600">API Offline</span>
          </div>
        ) : (
          <>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${health.data?.model_loaded ? "bg-green-500" : "bg-yellow-500"}`} />
              <span className="text-xs">{health.data?.model_loaded ? "Model Loaded" : "Demo Mode"}</span>
            </div>
            <div className="text-xs text-gray-500">
              Device: {health.data?.device} | Classes: {health.data?.diseases_count}
            </div>
          </>
        )}
      </div>

      {/* Settings */}
      <div className="space-y-4">
        <h3 className="text-xs font-semibold text-gray-500 uppercase">Settings</h3>

        <div>
          <label className="text-xs font-medium text-gray-700 flex justify-between">
            <span>Detection Threshold</span>
            <span className="text-teal-600 font-mono">{threshold.toFixed(2)}</span>
          </label>
          <input
            type="range" min={0.1} max={0.9} step={0.05}
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full mt-1 accent-teal-600"
            aria-label={`Detection threshold: ${threshold.toFixed(2)}`}
          />
          <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
            <span>Sensitive</span>
            <span>Specific</span>
          </div>
        </div>

        <div>
          <label className="text-xs font-medium text-gray-700 flex justify-between">
            <span>Top K Predictions</span>
            <span className="text-teal-600 font-mono">{topK}</span>
          </label>
          <input
            type="range" min={3} max={15} step={1}
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value))}
            className="w-full mt-1 accent-teal-600"
            aria-label={`Top K predictions: ${topK}`}
          />
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
        <p className="text-xs text-amber-700">
          <strong>Medical Disclaimer:</strong> AI screening tool for research purposes.
          Not a replacement for professional diagnosis.
        </p>
      </div>

      {/* Tech stack */}
      <div className="text-xs text-gray-400 space-y-1">
        <p>Next.js 16 + Zustand + TanStack Query</p>
        <p>FastAPI + PyTorch + ViGNN</p>
        <p>8x RTX A6000 | DDP Training</p>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop: always visible sidebar */}
      <aside className="hidden xl:block w-72 bg-white border-r h-full overflow-y-auto p-4 shrink-0">
        {content}
      </aside>

      {/* Mobile/Tablet: floating settings button + slide-over drawer */}
      <div className="xl:hidden">
        <button
          onClick={() => setOpen(true)}
          className="fixed bottom-20 lg:bottom-4 right-4 z-20 w-12 h-12 bg-teal-600 text-white rounded-full shadow-lg
                     flex items-center justify-center hover:bg-teal-700 active:bg-teal-800 transition-colors"
          aria-label="Open settings"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
          </svg>
        </button>

        {open && (
          <>
            <div className="mobile-overlay" onClick={() => setOpen(false)} aria-hidden="true" />
            <div className="fixed top-0 right-0 bottom-0 w-80 max-w-[85vw] bg-white z-50 shadow-2xl overflow-y-auto p-4 animate-slide-in">
              <div className="flex items-center justify-between mb-4">
                <h2 className="font-semibold text-slate-800">Settings</h2>
                <button
                  onClick={() => setOpen(false)}
                  className="w-8 h-8 flex items-center justify-center rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100"
                  aria-label="Close settings"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              {content}
            </div>
          </>
        )}
      </div>
    </>
  );
}
