import type { NextConfig } from "next";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8080";

const nextConfig: NextConfig = {
  output: "standalone",
  headers: async () => [
    {
      source: "/sw.js",
      headers: [
        { key: "Cache-Control", value: "no-cache, no-store, must-revalidate" },
        { key: "Service-Worker-Allowed", value: "/" },
      ],
    },
  ],
  // Rewrite /api/v1/* + /health* to the FastAPI backend so frontend and
  // backend appear same-origin to the browser. This makes auth cookies
  // first-party and lets proxy.ts read them.
  rewrites: async () => [
    { source: "/api/v1/:path*", destination: `${BACKEND_URL}/api/v1/:path*` },
    { source: "/health", destination: `${BACKEND_URL}/health` },
    { source: "/health/:path*", destination: `${BACKEND_URL}/health/:path*` },
  ],
};

export default nextConfig;
