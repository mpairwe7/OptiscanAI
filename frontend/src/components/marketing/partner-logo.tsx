/**
 * Inline-SVG wordmark lockups for partner organisations.
 *
 * Using SVG (not raster) so the logos render crisp at any DPR, ship zero
 * extra payload, and grayscale-on-hover cleanly. Swap any one for a real
 * SVG export from the partner's brand pack when available — just match
 * the bounding box (96×40 nominal).
 */
import type { CSSProperties } from "react";

interface Props {
  name: PartnerKey;
  className?: string;
  style?: CSSProperties;
  /** When true, render in full color (otherwise inherits via currentColor). */
  color?: boolean;
}

export type PartnerKey =
  | "makerere"
  | "mulago"
  | "ruharo"
  | "mengo"
  | "lubaga"
  | "idi"
  | "agakhan"
  | "makstartup"
  | "moh-uganda"
  | "crane-cloud";

export const PARTNER_LABELS: Record<PartnerKey, string> = {
  makerere: "Makerere University",
  mulago: "Mulago National Referral Hospital",
  ruharo: "Ruharo Eye Hospital",
  mengo: "Mengo Hospital",
  lubaga: "Lubaga Hospital",
  idi: "Infectious Diseases Institute",
  agakhan: "Aga Khan University Hospital",
  makstartup: "MakStartup",
  "moh-uganda": "Ministry of Health Uganda",
  "crane-cloud": "Crane Cloud",
};

export function PartnerLogo({ name, className = "", style, color = false }: Props) {
  const label = PARTNER_LABELS[name];
  return (
    <div
      className={`flex items-center gap-2 ${className}`}
      title={label}
      aria-label={label}
      style={style}
    >
      {renderMark(name, color)}
      <span className="font-bold tracking-tight text-[13px] sm:text-sm whitespace-nowrap">
        {WORDMARKS[name]}
      </span>
    </div>
  );
}

// Compact wordmarks chosen to be visually distinct in a logo cloud
const WORDMARKS: Record<PartnerKey, string> = {
  makerere: "Makerere",
  mulago: "Mulago",
  ruharo: "Ruharo Eye",
  mengo: "Mengo",
  lubaga: "Lubaga",
  idi: "IDI",
  agakhan: "Aga Khan",
  makstartup: "MakStartup",
  "moh-uganda": "MoH Uganda",
  "crane-cloud": "Crane Cloud",
};

function renderMark(name: PartnerKey, color: boolean) {
  // currentColor lets the parent grayscale group control the visual tone.
  const c = color ? undefined : "currentColor";
  switch (name) {
    case "makerere":
      // Shield silhouette — references Makerere's coat of arms motif.
      return (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
          <path
            d="M12 2L4 5v6c0 5 3.5 9.5 8 11 4.5-1.5 8-6 8-11V5l-8-3z"
            fill={c ?? "#0d9488"}
            stroke="currentColor"
            strokeOpacity="0.2"
          />
          <text x="12" y="15" textAnchor="middle" fontSize="9" fontWeight="800" fill="white">
            MAK
          </text>
        </svg>
      );
    case "mulago":
      // Medical cross inside a circle
      return (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
          <circle cx="12" cy="12" r="10" fill={c ?? "#dc2626"} />
          <rect x="10" y="5" width="4" height="14" fill="white" />
          <rect x="5" y="10" width="14" height="4" fill="white" />
        </svg>
      );
    case "ruharo":
      // Eye glyph
      return (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
          <path
            d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"
            stroke={c ?? "#0284c7"}
            strokeWidth="2"
            fill={c ? "white" : "currentColor"}
            fillOpacity={color ? 1 : 0.08}
          />
          <circle cx="12" cy="12" r="3" fill={c ?? "#0284c7"} />
          <circle cx="13" cy="11" r="0.8" fill="white" />
        </svg>
      );
    case "mengo":
      // Crown silhouette referencing the historic Mengo hospital crest
      return (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
          <path
            d="M3 17h18l-2-9-3 3-4-6-4 6-3-3-2 9z"
            fill={c ?? "#7c3aed"}
            stroke="currentColor"
            strokeOpacity="0.2"
          />
          <rect x="3" y="17" width="18" height="2.5" fill={c ?? "#7c3aed"} />
        </svg>
      );
    case "lubaga":
      // Trefoil (Catholic-mission lineage)
      return (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
          <circle cx="12" cy="7" r="4" fill={c ?? "#b91c1c"} />
          <circle cx="7" cy="15" r="4" fill={c ?? "#b91c1c"} />
          <circle cx="17" cy="15" r="4" fill={c ?? "#b91c1c"} />
          <circle cx="12" cy="13" r="2" fill="white" />
        </svg>
      );
    case "idi":
      // DNA double-helix simplified
      return (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
          <path
            d="M6 4c4 4 8 4 12 0M6 20c4-4 8-4 12 0M6 12c4 4 8 4 12 0"
            stroke={c ?? "#0891b2"}
            strokeWidth="2"
            strokeLinecap="round"
          />
        </svg>
      );
    case "agakhan":
      // Geometric quatrefoil — references AKU's brand language
      return (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
          <path
            d="M12 3L19 8l-3.5 4L19 16l-7 5-7-5 3.5-4L5 8z"
            fill={c ?? "#15803d"}
            stroke="currentColor"
            strokeOpacity="0.2"
          />
        </svg>
      );
    case "makstartup":
      // Stylised "M" lockup — wordmark-only style
      return (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
          <rect width="22" height="22" x="1" y="1" rx="6" fill={c ?? "#0f172a"} />
          <path d="M5 17V7l3.5 5 3.5-5v10M14 7l3.5 5 3.5-5v10" stroke="white" strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" fill="none" />
        </svg>
      );
    case "moh-uganda":
      // Heart + plus = MOH
      return (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
          <path
            d="M12 21s-7-4.5-7-10a4 4 0 017-2.6A4 4 0 0119 11c0 5.5-7 10-7 10z"
            fill={c ?? "#fbbf24"}
          />
          <rect x="10.5" y="7" width="3" height="8" fill="white" />
          <rect x="8" y="9.5" width="8" height="3" fill="white" />
        </svg>
      );
    case "crane-cloud":
      // Crane silhouette
      return (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
          <path
            d="M2 16c2 0 3 1 5 1s3-3 5-3 4 3 6 3 4-1 4-1"
            stroke={c ?? "#0ea5e9"}
            strokeWidth="2.2"
            strokeLinecap="round"
            fill="none"
          />
          <circle cx="20" cy="6" r="2" fill={c ?? "#0ea5e9"} />
          <path d="M20 8v4" stroke={c ?? "#0ea5e9"} strokeWidth="2" strokeLinecap="round" />
        </svg>
      );
  }
}
