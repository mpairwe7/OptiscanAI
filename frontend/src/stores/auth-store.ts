"use client";
import { create } from "zustand";

export interface MeUser {
  id: string;
  email: string;
  full_name: string | null;
  email_verified: boolean;
  is_superuser: boolean;
  organization: { id: string; name: string; slug: string; is_personal: boolean };
  role: string;
  subscription: {
    plan: { code: string; display_name: string; scan_limit_monthly: number | null; seat_limit: number | null };
    status: string;
    billing_cycle: string;
    current_period_start: string;
    current_period_end: string;
    cancel_at_period_end: boolean;
  } | null;
}

interface AuthState {
  user: MeUser | null;
  isHydrated: boolean;
  setUser: (u: MeUser | null) => void;
  setHydrated: (h: boolean) => void;
  signOut: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  isHydrated: false,
  setUser: (u) => set({ user: u }),
  setHydrated: (h) => set({ isHydrated: h }),
  signOut: () => set({ user: null }),
}));
