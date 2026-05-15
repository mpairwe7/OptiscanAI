# Frontend Setup

## Stack

| Component | Version | Purpose |
|---|---|---|
| Next.js | 16.2.3 | React framework with App Router |
| React | 19.2.4 | UI library |
| Bun | 1.3.12 | Package manager + runtime |
| Zustand | 5.x | Lightweight state management |
| TanStack Query | 5.x | Server state + API caching |
| Tailwind CSS | 4.x | Utility-first styling |
| TypeScript | 5.x | Type safety |

## Project Structure

```
frontend/
  src/
    app/
      layout.tsx           # Root layout with Providers
      page.tsx             # Main screening page (route-based navigation)
      globals.css          # Tailwind base styles
      manifest.ts          # PWA manifest generation
      favicon.ico          # App icon
    components/
      providers.tsx        # QueryClientProvider wrapper
      screening-page.tsx   # Main screening: upload + analyze + results
      dashboard-page.tsx   # Analytics dashboard: scan volumes, disease distribution
      review-page.tsx      # Human-in-the-loop review queue
      reports-page.tsx     # Prediction history and clinical reports
      system-page.tsx      # System health, model info, infrastructure
      image-upload.tsx     # Drag-drop image upload + analyze button
      results-panel.tsx    # Prediction results with confidence bars
      clinical-reasoning.tsx  # Knowledge graph clinical insights
      explainability-panel.tsx  # GradCAM/LIME/SHAP visualizations
      knowledge-graph-panel.tsx # Interactive disease relationship graph
      probability-chart.tsx     # Disease probability bar chart
      benchmark-panel.tsx       # Latency benchmarks display
      settings-sidebar.tsx      # Threshold slider, API status, model info
      nav-sidebar.tsx           # Main navigation sidebar
      sw-registrar.tsx          # Service worker registration (PWA)
      VoiceFirstChat.tsx        # Voice-first conversational UI with animated waveform
      VoiceVisionMode.tsx       # Camera preview + voice-vision multimodal mode
    stores/
      app-store.ts         # Zustand store (image, results, settings, navigation)
    hooks/
      use-predict.ts       # TanStack Query mutation for /api/v1/predict
      use-explain.ts       # TanStack Query mutation for /api/v1/explain/*
      useVoiceWebSocket.ts # WebSocket hook for real-time voice streaming
    stores/
      app-store.ts         # Zustand store (image, results, settings, navigation)
      voice-store.ts       # Zustand store (WebSocket state, audio, transcript, TTS)
    lib/
      api.ts               # API client (fetchHealth, predictImage, explain, etc.)
  public/
    logo.png               # App logo
    sw.js                  # Service worker (offline support)
    icon-192.png           # PWA icon (192x192)
    icon-512.png           # PWA icon (512x512)
    apple-touch-icon.png   # iOS home screen icon
  .env.production          # NEXT_PUBLIC_API_URL="" (relative, for HF Spaces nginx proxy)
  next.config.ts           # output: "standalone" (required for Docker deployment)
  tsconfig.json
  package.json
```

## Quick Start

```bash
# Install dependencies
cd frontend && bun install

# Start development server
bun dev
# or from project root:
make frontend
```

Frontend available at `http://localhost:3000`.

## Pages

The app uses Next.js 16 **route groups** to separate the public marketing
site from the authenticated app. All routes are real file routes — the
former Zustand-driven single-page setup has been retired.

### Public (`src/app/(public)/`)

| Route | Description |
|---|---|
| `/` | Marketing landing — hero (animated mesh + SVG fundus mockup), trust strip, logo cloud, how-it-works, value props, compliance badges, testimonials, pricing teaser, FAQ |
| `/pricing` | 4-card pricing grid with annual toggle (~17 % off) + full feature-comparison matrix |
| `/sign-in` · `/sign-up` · `/reset-password` · `/verify-email` | Auth flows (email + password, with magic-link fallback) |
| `/contact-sales` | Lead form for Health System tier |
| `/legal/privacy` · `/legal/terms` | Stub policies — PDP Act 2019 aware, marked "Draft pending legal review" |

### Authed (`src/app/(app)/app/`)

Gated by **`src/proxy.ts`** (note: this Next.js renames `middleware.ts` →
`proxy.ts`). Unauthenticated visitors are redirected to
`/sign-in?next=…`.

| Route | Description |
|---|---|
| `/app/dashboard` | Analytics: scan volumes, disease distribution, referral breakdown |
| `/app/screening` | Upload fundus image, run prediction, view results, explainability tabs |
| `/app/review` | Human-in-the-loop clinical review queue (Practice+) |
| `/app/reports` | Prediction history and exportable clinical reports |
| `/app/system` | Infrastructure health, model status, gate metrics |
| `/app/account` | Profile, sessions, sign-out |
| `/app/billing` | Plan card, invoices, change plan, **Downgrade to Free**, Stripe billing portal, inline renewal banner |
| `/app/usage` | Scan history chart, per-endpoint breakdown, CSV audit export |
| `/app/team` | Multi-seat management (Practice+): invite, role, seat purchases |
| `/app/checkout/[plan]` | Payment-method picker (Card / MTN / Airtel / Flutterwave) |
| `/app/checkout/success` | Post-checkout page that polls `/me` until the plan flips |
| `/app/admin/webhooks` | **Superuser only** — webhook event table with filters, JSON drawer, "Replay event" button |

Navigation lives in `src/components/nav-sidebar.tsx`, which uses
`usePathname` + `<Link>` (no more Zustand routing). The sidebar shows an
"Admin" section only when `useAuthStore.user.is_superuser === true`.

See [docs/23-billing-platform.md](23-billing-platform.md) § 15 for the
in-product billing UI surfaces (PaywallModal, UpsellSheet, UsageChip,
RenewalBanner, SeatManager, DowngradeDialog).

## State Management (Zustand)

```typescript
// stores/app-store.ts — full interface
useAppStore = create<AppState>({
  // Navigation
  currentPage: "dashboard",       // Page = "dashboard" | "screening" | "reports" | "review" | "system"
  sidebarCollapsed: false,
  mobileMenuOpen: false,

  // Image
  imageFile: File | null,
  imagePreview: string | null,

  // Prediction
  result: PredictionResponse | null,

  // Settings
  threshold: 0.5,
  topK: 5,

  // Explainability (5 XAI methods)
  gradcamResult: GradCAMResponse | null,
  limeResult: LIMEResponse | null,
  shapResult: SHAPResponse | null,
  igResult: IGResponse | null,
  eli5Result: ELI5Response | null,
  activeXaiMethod: string,

  // Session history
  scanHistory: { id, timestamp, result, imagePreview }[],

  // Actions: setPage, toggleSidebar, setImage, clearImage, setResult,
  //          setThreshold, setTopK, setGradcamResult, ..., clearExplainability,
  //          addScanToHistory
})
```

Zustand was chosen over Redux/Context for:
- Zero boilerplate
- No providers needed (except QueryClient)
- Direct store access in any component
- Tiny bundle size (~1KB)

## API Integration (TanStack Query)

```typescript
// hooks/use-predict.ts
const predict = useMutation({
  mutationFn: (file: File) => predictImage(file, threshold),
  onSuccess: (data) => setResult(data),
});

// Usage in component
predict.mutate(imageFile);
predict.isPending;  // Loading state
predict.isError;    // Error state
```

Health status polling:
```typescript
const health = useQuery({
  queryKey: ["health"],
  queryFn: fetchHealth,
  refetchInterval: 10_000,  // Poll every 10s
});
```

## API Client (`lib/api.ts`)

The API client exports 22+ typed functions organized by domain:

| Category | Functions |
|----------|----------|
| System | `fetchHealth()`, `fetchModelHealth()`, `fetchSystemInfo()`, `fetchAnalytics()` |
| Prediction | `predictImage(file, threshold)` |
| Diseases | `fetchDiseases()`, `fetchAllDiseaseInfo()`, `fetchDiseaseInfo(code)` |
| Knowledge Graph | `fetchKnowledgeGraph()` |
| Explainability | `explainGradCAM()`, `explainLIME()`, `explainSHAP()`, `explainIG()`, `explainELI5()`, `fetchAvailableMethods()` |
| Clinical | `explainReasoning(predictions)` |
| Review | `fetchPendingReviews()`, `fetchReviewStats()`, `resolveReview(id, decision)` |
| Agents | `fetchAgentStatus()`, `fetchAgentEvents()`, `fetchComplianceReport()` |

All functions use `NEXT_PUBLIC_API_URL` as the base URL. In HF Spaces (production), this is empty so calls use relative paths routed through nginx.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8080` | Backend API URL (development) |

In production (HF Spaces), `NEXT_PUBLIC_API_URL` is set to `""` (empty string) so API calls use relative paths — nginx on port 7860 routes `/api/*` to the backend.

## Building for Production

```bash
cd frontend && bun run build
# Output in frontend/.next/
# Standalone server in frontend/.next/standalone/server.js
```

The `output: "standalone"` setting in `next.config.ts` generates a self-contained Node.js server at `.next/standalone/server.js` that can run without the full `node_modules`. This is required for the Docker deployment.

## Deployment Modes

### Development (local)

```bash
# From project root - starts backend:8080 + frontend:3000
make dev
```

Backend and frontend run as separate processes. Frontend proxies API calls to `http://localhost:8080`.

### Docker — HF Spaces (production)

In the HF Spaces deployment (`Dockerfile.hf`), the frontend is built at Docker build time and served via the Next.js standalone server + nginx:

```
Build time:
  bun run build → .next/standalone/server.js + .next/static/*
  Static assets copied to /srv/nextjs/ for nginx direct serving

Runtime (supervisord):
  nginx (:7860) → routes static to filesystem, pages to Next.js, API to backend
  node server.js (:3000) → Next.js standalone SSR
  uvicorn (:8080) → FastAPI backend
```

**Static asset caching**: nginx serves `/_next/static/*` directly from `/srv/nextjs/` with 365-day cache headers and `Cache-Control: public, immutable`. This avoids hitting the Node.js process for static files.

**PWA support**: Service worker (`public/sw.js`) is registered via `SwRegistrar` component, enabling offline caching of static assets and the app shell.

### Docker — GPU backend only

When running `docker compose up -d api`, only the backend is containerized. Run the frontend separately with `make frontend` or deploy it to Vercel/Cloudflare Pages.

## Component Overview

### `ScreeningPage`
- Main workflow: upload → analyze → results
- Integrates ImageUpload, ResultsPanel, ExplainabilityPanel, ClinicalReasoning

### `ImageUpload`
- Drag-and-drop or click to upload
- Shows image preview
- Analyze button triggers TanStack Query mutation
- Loading/error states
- Handles 422 gate rejection errors with user-friendly message

### `ResultsPanel`
- Summary cards (detected count, inference time, referral priority)
- Ranked predictions with confidence bars (color-coded by severity)
- Fundus gate confidence badge (green/amber/red)
- Clinical disclaimer

### `ExplainabilityPanel`
- GradCAM heatmap overlay
- LIME superpixel importance
- SHAP feature importance
- Integrated Gradients attribution
- Tabbed interface for switching methods

### `ClinicalReasoning`
- Knowledge graph-based disease co-occurrence analysis
- Refined predictions with boost/suppress explanations
- Treatment recommendations and referral priority

### `KnowledgeGraphPanel`
- Interactive visualization of 45-disease relationship graph
- Disease nodes colored by category (Vascular, Degenerative, etc.)
- Co-occurrence edges with relationship strength

### `DashboardPage`
- Scan volume trends (daily/weekly)
- Disease distribution charts
- Referral priority breakdown
- Inference latency metrics

### `ReviewPage`
- Pending clinical reviews with priority filtering
- Resolve/escalate actions with clinician notes

### `SystemPage`
- API health status
- Model info (architecture, device, diseases)
- Gate status and metrics
- Infrastructure details

### `SettingsSidebar`
- Live API health status with polling
- Threshold slider (0.1-0.9)
- Top-K predictions slider
- Model/GPU info
- Medical disclaimer

### `VoiceFirstChat` (Phase 5)
- Voice-first conversational screening UI
- Animated waveform canvas visualization
- Microphone input capture + speaker output
- Real-time ASR transcript display
- Integrates with voice WebSocket pipeline

### `VoiceVisionMode` (Phase 5)
- Camera preview with live capture
- Voice narration overlay
- Combined voice + vision multimodal screening

## Voice WebSocket Hook (`useVoiceWebSocket`)

```typescript
// hooks/useVoiceWebSocket.ts
const { connect, disconnect, sendAudio, isConnected } = useVoiceWebSocket({
  url: "ws://localhost:8080/v1/voice/stream",
  onTranscript: (text) => ...,
  onTTSAudio: (audioBlob) => ...,
  onResult: (screeningResult) => ...,
});
```

- Exponential-backoff reconnect (1s – 30s)
- Captures microphone audio as 250ms WebM/Opus chunks
- Dispatches transcripts and TTS audio to voice store

## Voice Store (Zustand)

```typescript
// stores/voice-store.ts
useVoiceStore = create<VoiceState>({
  isConnected: false,
  isRecording: false,
  transcript: "",
  audioChunks: Blob[],
  ttsPlaying: false,
  waveformData: Float32Array,
  // Actions: startRecording, stopRecording, setTranscript, playTTS, ...
})
```
