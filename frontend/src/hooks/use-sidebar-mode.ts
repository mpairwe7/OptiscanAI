"use client";
import { useSyncExternalStore } from "react";
import {
  getSidebarMode,
  setSidebarMode,
  subscribeSidebarMode,
  type SidebarMode,
} from "@/lib/sidebar-mode";

/**
 * Read and set the navigation rail's mode.
 *
 * The server snapshot is always "hover" — the server cannot know a viewer's
 * stored preference, and claiming one would make the markup disagree with what
 * the pre-paint script already stamped on <html>. Hydration therefore matches
 * the default, and a pinned rail is correct from the first frame because its
 * width comes from the `data-rail-mode` attribute in CSS, not from this value.
 */
export function useSidebarMode(): [SidebarMode, (mode: SidebarMode) => void] {
  const mode = useSyncExternalStore(subscribeSidebarMode, getSidebarMode, () => "hover" as const);
  return [mode, setSidebarMode];
}
