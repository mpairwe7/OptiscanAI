// RetinalAI Service Worker - Offline support + caching
const CACHE_NAME = "retinalai-v1";
const PRECACHE = ["/", "/icon-192.png", "/icon-512.png", "/logo.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const { request } = e;
  // Network-first for API calls
  if (request.url.includes("/api/") || request.url.includes("/health")) {
    e.respondWith(fetch(request).catch(() => caches.match(request)));
    return;
  }
  // Cache-first for static assets
  e.respondWith(
    caches.match(request).then((cached) => cached || fetch(request).then((res) => {
      if (res.ok && request.method === "GET") {
        const clone = res.clone();
        caches.open(CACHE_NAME).then((c) => c.put(request, clone));
      }
      return res;
    }))
  );
});
