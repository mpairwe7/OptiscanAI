/**
 * Pure-SVG fundus mockup with Grad-CAM-style heatmap overlay + detection chip.
 * Used in the hero. No external image dependency — keeps LCP low and looks
 * crisp at any DPR.
 */
export function FundusMockup({ className = "" }: { className?: string }) {
  return (
    <div className={`relative ${className}`}>
      <div className="aspect-[4/3] rounded-2xl overflow-hidden shadow-2xl ring-1 ring-slate-900/10 bg-slate-900">
        <svg
          viewBox="0 0 800 600"
          xmlns="http://www.w3.org/2000/svg"
          className="w-full h-full"
          aria-label="Annotated retinal fundus image"
          role="img"
        >
          <defs>
            <radialGradient id="fundusBg" cx="50%" cy="50%" r="60%">
              <stop offset="0%" stopColor="#3a1502" />
              <stop offset="40%" stopColor="#7a2104" />
              <stop offset="80%" stopColor="#3a0a01" />
              <stop offset="100%" stopColor="#1a0500" />
            </radialGradient>
            <radialGradient id="opticDisc" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#ffe6a8" />
              <stop offset="60%" stopColor="#f0b15a" />
              <stop offset="100%" stopColor="#a25a18" />
            </radialGradient>
            <radialGradient id="macula" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#2a0801" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#2a0801" stopOpacity="0" />
            </radialGradient>
            <radialGradient id="heat1" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#ff3b30" stopOpacity="0.7" />
              <stop offset="60%" stopColor="#ff9500" stopOpacity="0.35" />
              <stop offset="100%" stopColor="#ff9500" stopOpacity="0" />
            </radialGradient>
            <radialGradient id="heat2" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#ffcc00" stopOpacity="0.55" />
              <stop offset="100%" stopColor="#ffcc00" stopOpacity="0" />
            </radialGradient>
          </defs>

          {/* Eye sclera */}
          <rect width="800" height="600" fill="#08080a" />
          {/* Retinal background — circular fundus */}
          <circle cx="400" cy="300" r="290" fill="url(#fundusBg)" />

          {/* Vessels — branching outward */}
          <g
            stroke="#5a1308"
            fill="none"
            strokeLinecap="round"
            opacity="0.85"
          >
            {/* main arcades */}
            <path d="M 480 280 Q 580 220 700 200" strokeWidth="6" />
            <path d="M 480 320 Q 580 380 700 410" strokeWidth="6" />
            <path d="M 460 280 Q 360 220 200 200" strokeWidth="5" />
            <path d="M 460 320 Q 360 380 200 410" strokeWidth="5" />
            {/* branches */}
            <path d="M 600 230 Q 640 200 680 170" strokeWidth="3" />
            <path d="M 600 370 Q 640 410 680 440" strokeWidth="3" />
            <path d="M 270 230 Q 230 200 190 170" strokeWidth="3" />
            <path d="M 270 370 Q 230 410 190 440" strokeWidth="3" />
            <path d="M 530 250 Q 550 200 540 150" strokeWidth="3" />
            <path d="M 540 350 Q 560 400 550 450" strokeWidth="3" />
            <path d="M 410 240 Q 430 180 420 130" strokeWidth="3" />
            <path d="M 410 360 Q 430 420 420 470" strokeWidth="3" />
            {/* thin twigs */}
            <path d="M 660 195 Q 700 180 740 165" strokeWidth="2" opacity="0.7" />
            <path d="M 660 405 Q 700 420 740 435" strokeWidth="2" opacity="0.7" />
          </g>

          {/* Optic disc */}
          <ellipse cx="475" cy="300" rx="34" ry="36" fill="url(#opticDisc)" />
          <ellipse cx="475" cy="300" rx="14" ry="16" fill="#fff5d4" opacity="0.85" />

          {/* Macula (dark central spot) */}
          <ellipse cx="320" cy="312" rx="55" ry="42" fill="url(#macula)" />

          {/* Subtle micro-aneurysms */}
          <g fill="#7a0d05">
            <circle cx="340" cy="280" r="3" />
            <circle cx="355" cy="330" r="2.5" />
            <circle cx="290" cy="340" r="3.5" />
            <circle cx="300" cy="270" r="2" />
            <circle cx="380" cy="350" r="2.5" />
          </g>

          {/* Grad-CAM heatmap overlay — focused on macula (typical DR pattern) */}
          <g style={{ mixBlendMode: "screen" }}>
            <circle cx="320" cy="312" r="120" fill="url(#heat1)" />
            <circle cx="320" cy="312" r="180" fill="url(#heat2)" />
            <circle cx="475" cy="300" r="60" fill="url(#heat2)" opacity="0.6" />
          </g>

          {/* Corner brand mark */}
          <g transform="translate(20 20)" opacity="0.45">
            <rect width="120" height="22" rx="11" fill="#0d9488" />
            <text
              x="60"
              y="15"
              textAnchor="middle"
              fontSize="11"
              fontFamily="ui-sans-serif, system-ui"
              fontWeight="700"
              fill="white"
              letterSpacing="0.05em"
            >
              GRAD-CAM
            </text>
          </g>
        </svg>
      </div>

      {/* Detection chip (floats lower-left) */}
      <div className="absolute -bottom-5 -left-3 sm:-left-5 bg-white border border-slate-200 rounded-xl shadow-xl p-3.5 sm:p-4 max-w-[230px]">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse-dot" />
          <span className="text-[10px] uppercase tracking-wider font-bold text-slate-500">
            Detected · HIGH priority
          </span>
        </div>
        <div className="mt-1 font-semibold text-slate-900 text-sm">
          Diabetic Retinopathy
        </div>
        <div className="mt-2 flex items-center justify-between text-xs">
          <span className="text-slate-500">Confidence</span>
          <span className="font-mono font-semibold text-slate-900">0.94</span>
        </div>
        <div className="mt-1.5 h-1 bg-slate-100 rounded-full overflow-hidden">
          <div className="h-full bg-amber-500" style={{ width: "94%" }} />
        </div>
      </div>

      {/* Latency chip (floats upper-right) */}
      <div className="absolute -top-3 -right-3 sm:-right-5 bg-slate-900 text-white rounded-xl shadow-xl px-3 py-2 text-xs font-mono hidden sm:block">
        <span className="text-emerald-400">●</span> 85 ms inference
      </div>
    </div>
  );
}
