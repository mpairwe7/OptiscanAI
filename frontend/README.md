# OptiscanAI Frontend

The Next.js 16 web app for OptiscanAI — clinician dashboard, fundus screening,
human-in-the-loop review, explainability, billing, and a voice-first mode.

> ⚠️ This is a **customized** Next.js — read [`AGENTS.md`](AGENTS.md) before
> changing framework-level code.

## Stack

- **Next.js 16** (App Router) · **React 19** · **TypeScript** (strict)
- **Bun** (package manager / runtime) · **Tailwind CSS v4**
- **Zustand** (state) · **TanStack Query** (data fetching)

## Setup

```bash
bun install
bun dev            # dev server on http://localhost:3000
```

Or from the repo root: `make frontend` (frontend only) / `make dev` (backend + frontend).

## Build

```bash
bun run build      # production build (.next/standalone)
bun run start      # serve the production build
bun run lint       # ESLint
```

In Docker the app is built to a standalone server and served behind nginx (see
[`../deploy/`](../deploy/README.md)).

## Environment

`NEXT_PUBLIC_*` variables are **baked into the browser bundle at build time** —
never put secrets here.

| Variable | Dev (`.env.local`) | Prod (`.env.production`) |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8080` | `https://www.optiscan.makstartup.com` |

In the full-stack Docker image this is left empty and nginx proxies `/api` to the
backend. See [`../docs/24-environment-variables.md`](../docs/24-environment-variables.md).

## Structure

```
src/
├── app/          # App Router routes + layouts (dashboard, screening, review, auth, billing)
├── components/   # UI + feature panels (results, explainability, knowledge graph, voice)
├── stores/       # Zustand stores (auth, app, billing, voice)
├── hooks/        # Custom React hooks
└── lib/          # API client, formatters, plan definitions
```

More detail: [`../docs/07-frontend-setup.md`](../docs/07-frontend-setup.md).
