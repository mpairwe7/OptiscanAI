import type { Metadata, Viewport } from "next";
import { headers } from "next/headers";
import { Geist, Geist_Mono } from "next/font/google";
import { Providers } from "@/components/providers";
import { ServiceWorkerRegistrar } from "@/components/sw-registrar";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const CANONICAL_ORIGIN = "https://www.optiscan.makstartup.com";

export async function generateMetadata(): Promise<Metadata> {
  // OpenGraph + Twitter image URLs must be absolute AND reachable, otherwise
  // WhatsApp/Telegram/FB previews silently drop the image. Build the origin
  // from the current request so previews work on whichever host is live —
  // Crane Cloud's generated URL today, the canonical domain once DNS lands.
  // `alternates.canonical` still pins SEO to the canonical origin separately.
  const hdrs = await headers();
  const host = hdrs.get("x-forwarded-host") || hdrs.get("host") || "";
  const isLocal = host.includes("localhost") || host.includes("127.0.0.1");
  const proto = isLocal ? "http" : "https";
  const origin = host ? `${proto}://${host}` : CANONICAL_ORIGIN;

  return {
    metadataBase: new URL(origin),
    title: {
      default: "OptiscanAI — Clinical Retinal Screening, Explained",
      template: "%s · OptiscanAI",
    },
    description:
      "AI-powered multi-disease retinal screening with explainable AI and clinical knowledge graph reasoning. 45 diseases. Built for Ugandan healthcare.",
    manifest: "/manifest.webmanifest",
    alternates: { canonical: CANONICAL_ORIGIN },
    keywords: [
      "retinal screening",
      "diabetic retinopathy",
      "AI eye screening",
      "explainable AI healthcare",
      "Uganda healthcare AI",
      "fundus image analysis",
    ],
    authors: [{ name: "MakStartup" }],
    openGraph: {
      title: "OptiscanAI — Clinical Retinal Screening, Explained",
      description:
        "Detect 45 retinal diseases with explainable AI. Knowledge-graph clinical reasoning. Built for Ugandan healthcare.",
      url: origin,
      siteName: "OptiscanAI",
      locale: "en_UG",
      type: "website",
      images: [
        {
          url: `${origin}/og-image.png`,
          width: 1200,
          height: 630,
          alt: "OptiscanAI - Clinical Retinal Screening Platform",
          type: "image/png",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title: "OptiscanAI — Clinical Retinal Screening, Explained",
      description: "AI-powered multi-disease eye screening with explainable AI. Built for Ugandan healthcare.",
      images: [
        {
          url: `${origin}/og-image.png`,
          width: 1200,
          height: 630,
          alt: "OptiscanAI - Clinical Retinal Screening Platform",
        },
      ],
    },
    robots: { index: true, follow: true, googleBot: { index: true, follow: true } },
    appleWebApp: {
      capable: true,
      statusBarStyle: "black-translucent",
      title: "OptiscanAI",
    },
    icons: {
      icon: [
        { url: "/favicon-16x16.png", sizes: "16x16", type: "image/png" },
        { url: "/favicon-32x32.png", sizes: "32x32", type: "image/png" },
        { url: "/favicon-48x48.png", sizes: "48x48", type: "image/png" },
      ],
      apple: [{ url: "/apple-touch-icon.png", sizes: "180x180" }],
    },
    other: {
      "mobile-web-app-capable": "yes",
    },
  };
}

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#0f172a" },
    { media: "(prefers-color-scheme: dark)", color: "#0f172a" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="h-full">
        <a href="#main-content" className="skip-to-main">
          Skip to main content
        </a>
        <Providers>{children}</Providers>
        <ServiceWorkerRegistrar />
      </body>
    </html>
  );
}
