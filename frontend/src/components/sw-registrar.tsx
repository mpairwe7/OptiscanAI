"use client";
import { useEffect } from "react";

/**
 * Registers /sw.js on production builds. Bumps the SW immediately when a new
 * version is detected so users always get the latest offline shell without a
 * manual reload. Dev builds skip registration to avoid HMR conflicts.
 */
export function ServiceWorkerRegistrar() {
  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!("serviceWorker" in navigator)) return;
    if (process.env.NODE_ENV !== "production") return;

    let cancelled = false;

    navigator.serviceWorker
      .register("/sw.js", { scope: "/" })
      .then((registration) => {
        if (cancelled) return;
        // Check for update on every page load.
        registration.update().catch(() => {});

        // If a waiting worker exists, prompt it to skip waiting.
        if (registration.waiting) {
          registration.waiting.postMessage({ type: "SKIP_WAITING" });
        }

        registration.addEventListener("updatefound", () => {
          const next = registration.installing;
          if (!next) return;
          next.addEventListener("statechange", () => {
            if (next.state === "installed" && navigator.serviceWorker.controller) {
              // New SW installed; activate immediately.
              next.postMessage({ type: "SKIP_WAITING" });
            }
          });
        });
      })
      .catch((err) => {
        // Don't surface to user — SW is enhancement, not requirement.
        if (process.env.NODE_ENV !== "production") {
          console.warn("SW registration failed:", err);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);
  return null;
}
