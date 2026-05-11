import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "OptiscanAI - Clinical Screening Platform",
    short_name: "OptiscanAI",
    description:
      "AI-powered multi-disease retinal screening with clinical knowledge graph reasoning. 45 diseases. Explainable AI.",
    start_url: "/",
    display: "standalone",
    orientation: "any",
    background_color: "#0f172a",
    theme_color: "#0f172a",
    categories: ["medical", "health", "productivity"],
    icons: [
      { src: "/favicon-48x48.png", sizes: "48x48", type: "image/png" },
      { src: "/icon-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
      { src: "/apple-touch-icon.png", sizes: "180x180", type: "image/png" },
    ],
    screenshots: [],
    prefer_related_applications: false,
  };
}
