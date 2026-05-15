"use client";
/**
 * apiFetch — wraps fetch() with:
 *  - credentials: "include" so httpOnly auth cookies are sent
 *  - one auto-retry on 401 via /api/auth/refresh
 *  - opens the paywall on 402 with quota_exceeded
 *  - opens the upsell sheet on 403 with feature_locked
 */
import { useBillingStore, type FeatureLockedPayload, type QuotaPayload } from "@/stores/billing-store";
import { useAuthStore } from "@/stores/auth-store";

export interface ApiFetchOptions extends RequestInit {
  /** When true, do not auto-trigger the paywall/upsell (used for benign GETs that may 402). */
  silent?: boolean;
}

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, message: string, body: unknown) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function callRefresh(): Promise<boolean> {
  try {
    const r = await fetch("/api/auth/refresh", {
      method: "POST",
      credentials: "include",
    });
    return r.ok;
  } catch {
    return false;
  }
}

export async function apiFetch(input: RequestInfo | URL, init?: ApiFetchOptions): Promise<Response> {
  const { silent, ...rest } = init ?? {};
  const opts: RequestInit = {
    credentials: "include",
    ...rest,
  };
  let res = await fetch(input, opts);

  // Single retry on 401 with refresh
  if (res.status === 401) {
    const refreshed = await callRefresh();
    if (refreshed) {
      res = await fetch(input, opts);
    } else if (!silent) {
      useAuthStore.getState().signOut();
      if (typeof window !== "undefined" && !window.location.pathname.startsWith("/sign-in")) {
        const next = encodeURIComponent(window.location.pathname + window.location.search);
        window.location.href = `/sign-in?next=${next}`;
      }
    }
  }

  if (res.status === 402) {
    const body = await res.clone().json().catch(() => null) as { detail?: QuotaPayload } | null;
    const payload = body?.detail;
    if (!silent && payload?.error === "quota_exceeded") {
      useBillingStore.getState().openPaywall(payload);
    }
  }

  if (res.status === 403) {
    const body = await res.clone().json().catch(() => null) as { detail?: FeatureLockedPayload } | null;
    const payload = body?.detail;
    if (!silent && payload?.error === "feature_locked") {
      useBillingStore.getState().openUpsell(payload);
    }
  }

  return res;
}

export async function apiJson<T>(input: RequestInfo | URL, init?: ApiFetchOptions): Promise<T> {
  const res = await apiFetch(input, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = body?.detail;
    const message = typeof detail === "string" ? detail : body?.message ?? res.statusText;
    throw new ApiError(res.status, message, body);
  }
  return (await res.json()) as T;
}
