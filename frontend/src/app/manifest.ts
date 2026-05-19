import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "OptiscanAI — Clinical Retinal Screening",
    short_name: "OptiscanAI",
    description:
      "AI-powered multi-disease retinal screening with clinical knowledge-graph reasoning. 45 diseases. Explainable AI. Built for Ugandan healthcare.",
    id: "/?source=pwa",
    start_url: "/?source=pwa",
    scope: "/",
    display: "standalone",
    display_override: ["standalone", "minimal-ui", "browser"],
    orientation: "portrait-primary",
    background_color: "#ffffff",
    theme_color: "#0f172a",
    lang: "en-UG",
    dir: "ltr",
    categories: ["medical", "health", "productivity"],
    icons: [
      { src: "/favicon-16x16.png", sizes: "16x16", type: "image/png" },
      { src: "/favicon-32x32.png", sizes: "32x32", type: "image/png" },
      { src: "/favicon-48x48.png", sizes: "48x48", type: "image/png" },
      { src: "/apple-touch-icon.png", sizes: "180x180", type: "image/png" },
      { src: "/icon-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
      { src: "/icon-192.png", sizes: "192x192", type: "image/png", purpose: "maskable" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png", purpose: "any" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
    shortcuts: [
      {
        name: "Start screening",
        short_name: "Screen",
        description: "Upload a fundus image and run retinal screening",
        url: "/app/screening?source=pwa-shortcut",
        icons: [{ src: "/icon-192.png", sizes: "192x192", type: "image/png" }],
      },
      {
        name: "Pricing",
        short_name: "Pricing",
        description: "View plans and subscription pricing",
        url: "/pricing?source=pwa-shortcut",
        icons: [{ src: "/icon-192.png", sizes: "192x192", type: "image/png" }],
      },
      {
        name: "Sign in",
        short_name: "Sign in",
        description: "Sign in to your OptiscanAI account",
        url: "/sign-in?source=pwa-shortcut",
        icons: [{ src: "/icon-192.png", sizes: "192x192", type: "image/png" }],
      },
    ],
    screenshots: [],
    prefer_related_applications: false,
  };
}
