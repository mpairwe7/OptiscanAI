"use client";
import { create } from "zustand";

export interface QuotaPayload {
  error: "quota_exceeded";
  message: string;
  plan: { code: string; scan_limit_monthly: number };
  usage: { used: number; limit: number; resets_at: string };
  upgrade_url: string;
  recommended_plan: string;
}

export interface FeatureLockedPayload {
  error: "feature_locked";
  message: string;
  feature: string;
  required_plan: string;
  current_plan: string;
  upgrade_url: string;
}

interface BillingState {
  paywallOpen: boolean;
  paywallPayload: QuotaPayload | null;
  upsellOpen: boolean;
  upsellPayload: FeatureLockedPayload | null;

  openPaywall: (p: QuotaPayload) => void;
  closePaywall: () => void;
  openUpsell: (p: FeatureLockedPayload) => void;
  closeUpsell: () => void;
}

export const useBillingStore = create<BillingState>((set) => ({
  paywallOpen: false,
  paywallPayload: null,
  upsellOpen: false,
  upsellPayload: null,
  openPaywall: (p) => set({ paywallOpen: true, paywallPayload: p }),
  closePaywall: () => set({ paywallOpen: false }),
  openUpsell: (p) => set({ upsellOpen: true, upsellPayload: p }),
  closeUpsell: () => set({ upsellOpen: false }),
}));
