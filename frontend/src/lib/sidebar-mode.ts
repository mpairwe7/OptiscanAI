/**
 * Whether the navigation rail stays pinned open or expands on hover.
 *
 * Two modes rather than a plain open/closed toggle, because the rail is useful
 * in both states and different people want different things from it: a
 * technician working the review queue all day wants the labels pinned, while a
 * clinician dipping in to read one report wants the screen back. Hover is the
 * default — the console's own pages are wide tables (screening results, the
 * review queue) and 224px of permanent chrome pushes columns off a laptop.
 *
 * State lives outside React on purpose. A pre-paint inline script stamps the
 * root element so a pinned rail is already full width in the first frame, and
 * the React side reads it through useSyncExternalStore. Reading localStorage in
 * an effect instead would animate the rail open on every load for pinned users.
 */
export type SidebarMode = "hover" | "always-open";

const KEY = "optiscan.sidebarMode";
export const SIDEBAR_EVENT = "optiscan:sidebar-mode";

export function getSidebarMode(): SidebarMode {
  if (typeof window === "undefined") return "hover";
  try {
    return window.localStorage.getItem(KEY) === "always-open" ? "always-open" : "hover";
  } catch {
    // Private mode / storage disabled. The preference is a nicety, not a
    // requirement, so fall back to the default rather than breaking the shell.
    return "hover";
  }
}

export function setSidebarMode(mode: SidebarMode): void {
  if (typeof window === "undefined") return;
  try {
    if (mode === "always-open") window.localStorage.setItem(KEY, mode);
    else window.localStorage.removeItem(KEY);
  } catch {
    // Same reasoning as getSidebarMode — the rail still works this session.
  }
  document.documentElement.dataset.railMode = mode;
  window.dispatchEvent(new CustomEvent(SIDEBAR_EVENT));
}

/** Subscribe to mode changes — the store half of useSyncExternalStore. */
export function subscribeSidebarMode(onChange: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  window.addEventListener(SIDEBAR_EVENT, onChange);
  // `storage` fires only in *other* tabs, which is exactly what it is for here:
  // pinning the rail in one tab should not leave a second tab disagreeing.
  window.addEventListener("storage", onChange);
  return () => {
    window.removeEventListener(SIDEBAR_EVENT, onChange);
    window.removeEventListener("storage", onChange);
  };
}

/**
 * Evaluated before paint in the document head.
 *
 * Kept as a string rather than a function so it can be inlined verbatim; it
 * must not reference anything outside its own scope. Wrapped in try/catch
 * because localStorage throws outright when site data is blocked, and an
 * uncaught error here would break hydration for the whole app.
 */
export const SIDEBAR_INIT_SCRIPT =
  "(function(){try{var m=localStorage.getItem('optiscan.sidebarMode');" +
  "document.documentElement.dataset.railMode=(m==='always-open')?'always-open':'hover';}" +
  "catch(e){document.documentElement.dataset.railMode='hover';}})();";
