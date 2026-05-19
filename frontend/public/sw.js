// OptiscanAI Service Worker — offline support + PWA caching
// Bump CACHE_NAME any time you change the cached assets or fetch logic.
const CACHE_NAME = "optiscanai-v2";
const OFFLINE_URL = "/offline.html";

// Minimal precache — just the shell needed to render the offline fallback
// and the app icons. Real page content is cached lazily on first visit.
const PRECACHE = [
  OFFLINE_URL,
  "/icon-192.png",
  "/icon-512.png",
  "/logo.png",
  "/favicon-32x32.png",
  "/manifest.webmanifest",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      await cache.addAll(PRECACHE);
      // Activate immediately so a fresh SW takes over.
      self.skipWaiting();
    })()
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      // Enable navigation preload — speeds up first paint for navigations.
      if ("navigationPreload" in self.registration) {
        try {
          await self.registration.navigationPreload.enable();
        } catch {
          /* ignore */
        }
      }
      // Drop old caches from previous versions.
      const keys = await caches.keys();
      await Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)));
      await self.clients.claim();
    })()
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;

  // Only handle GET. Mutations should always hit the network.
  if (request.method !== "GET") return;

  const url = new URL(request.url);

  // Skip cross-origin (CDN, fonts, etc.) so we don't poison the cache.
  if (url.origin !== self.location.origin) return;

  // Network-first for API + health probes. On failure return cached if available.
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/health")) {
    event.respondWith(networkFirst(request));
    return;
  }

  // Navigation requests: network-first with offline fallback.
  if (request.mode === "navigate") {
    event.respondWith(navigationStrategy(event));
    return;
  }

  // Static assets: stale-while-revalidate.
  event.respondWith(staleWhileRevalidate(request));
});

async function networkFirst(request) {
  try {
    const fresh = await fetch(request);
    if (fresh.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, fresh.clone());
    }
    return fresh;
  } catch {
    const cached = await caches.match(request);
    if (cached) return cached;
    return new Response(JSON.stringify({ error: "offline" }), {
      status: 503,
      headers: { "Content-Type": "application/json" },
    });
  }
}

async function navigationStrategy(event) {
  try {
    // Use preloaded response if available (faster than re-fetching).
    const preload = await event.preloadResponse;
    if (preload) return preload;
    const fresh = await fetch(event.request);
    if (fresh.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(event.request, fresh.clone());
    }
    return fresh;
  } catch {
    const cached = await caches.match(event.request);
    if (cached) return cached;
    const offline = await caches.match(OFFLINE_URL);
    return (
      offline ||
      new Response("Offline", { status: 503, headers: { "Content-Type": "text/plain" } })
    );
  }
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  const networkPromise = fetch(request)
    .then((response) => {
      if (response.ok) cache.put(request, response.clone());
      return response;
    })
    .catch(() => cached);
  return cached || networkPromise;
}

// Allow page to trigger SW update via postMessage.
self.addEventListener("message", (event) => {
  if (event.data?.type === "SKIP_WAITING") self.skipWaiting();
});
