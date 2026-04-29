# Frontend Setup

## Stack

| Component | Version | Purpose |
|---|---|---|
| Next.js | 16.2.3 | React framework with App Router |
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
      page.tsx             # Main screening page
      globals.css          # Tailwind base styles
    components/
      providers.tsx        # QueryClientProvider wrapper
      image-upload.tsx     # Drag-drop image upload + analyze button
      results-panel.tsx    # Prediction results with confidence bars
      settings-sidebar.tsx # Threshold slider, API status, model info
    stores/
      app-store.ts         # Zustand store (image, results, settings)
    hooks/
      use-predict.ts       # TanStack Query mutation for /api/v1/predict
    lib/
      api.ts               # API client (fetchHealth, predictImage, etc.)
  .env.local               # NEXT_PUBLIC_API_URL=http://localhost:8080
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

## State Management (Zustand)

```typescript
// stores/app-store.ts
useAppStore = create({
  imageFile: File | null,
  imagePreview: string | null,
  result: PredictionResponse | null,
  threshold: 0.5,
  topK: 5,
  // ... actions: setImage, clearImage, setResult, setThreshold
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

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8080` | Backend API URL |

## Building for Production

```bash
cd frontend && bun run build
# Output in frontend/.next/
```

## Running Both (Development)

```bash
# From project root - starts backend:8080 + frontend:3000
make dev
```

## Component Overview

### `ImageUpload`
- Drag-and-drop or click to upload
- Shows image preview
- Analyze button triggers TanStack Query mutation
- Loading/error states

### `ResultsPanel`
- Summary cards (detected count, inference time, referral priority)
- Ranked predictions with confidence bars (color-coded by severity)
- Clinical disclaimer

### `SettingsSidebar`
- Live API health status with polling
- Threshold slider (0.1-0.9)
- Top-K predictions slider
- Model/GPU info
- Medical disclaimer
