import { create } from "zustand";
import type { PredictionResponse, GradCAMResponse, LIMEResponse, SHAPResponse, IGResponse, ELI5Response } from "@/lib/api";

export type Page = "dashboard" | "screening" | "reports" | "review" | "system";

interface AppState {
  // Navigation
  currentPage: Page;
  setPage: (p: Page) => void;
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
  mobileMenuOpen: boolean;
  setMobileMenuOpen: (open: boolean) => void;

  // Image
  imageFile: File | null;
  imagePreview: string | null;
  setImage: (file: File) => void;
  clearImage: () => void;

  // Prediction
  result: PredictionResponse | null;
  setResult: (r: PredictionResponse) => void;
  clearResult: () => void;

  // Settings
  threshold: number;
  setThreshold: (t: number) => void;
  topK: number;
  setTopK: (k: number) => void;

  // Explainability
  gradcamResult: GradCAMResponse | null;
  setGradcamResult: (r: GradCAMResponse | null) => void;
  limeResult: LIMEResponse | null;
  setLimeResult: (r: LIMEResponse | null) => void;
  shapResult: SHAPResponse | null;
  setShapResult: (r: SHAPResponse | null) => void;
  igResult: IGResponse | null;
  setIgResult: (r: IGResponse | null) => void;
  eli5Result: ELI5Response | null;
  setEli5Result: (r: ELI5Response | null) => void;
  activeXaiMethod: string;
  setActiveXaiMethod: (m: string) => void;
  clearExplainability: () => void;

  // Screening history (session-only)
  scanHistory: { id: string; timestamp: string; result: PredictionResponse; imagePreview: string }[];
  addScanToHistory: (id: string, result: PredictionResponse, imagePreview: string) => void;
}

export const useAppStore = create<AppState>((set) => ({
  currentPage: "dashboard",
  setPage: (p) => set({ currentPage: p, mobileMenuOpen: false }),
  sidebarCollapsed: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
  mobileMenuOpen: false,
  setMobileMenuOpen: (open) => set({ mobileMenuOpen: open }),

  imageFile: null,
  imagePreview: null,
  setImage: (file) =>
    set({
      imageFile: file,
      imagePreview: URL.createObjectURL(file),
      result: null,
      gradcamResult: null,
      limeResult: null,
      shapResult: null,
      igResult: null,
      eli5Result: null,
    }),
  clearImage: () =>
    set((s) => {
      if (s.imagePreview) URL.revokeObjectURL(s.imagePreview);
      return {
        imageFile: null,
        imagePreview: null,
        result: null,
        gradcamResult: null,
        limeResult: null,
        shapResult: null,
        igResult: null,
        eli5Result: null,
      };
    }),

  result: null,
  setResult: (r) => set({ result: r }),
  clearResult: () => set({ result: null }),

  threshold: 0.5,
  setThreshold: (t) => set({ threshold: t }),
  topK: 5,
  setTopK: (k) => set({ topK: k }),

  gradcamResult: null,
  setGradcamResult: (r) => set({ gradcamResult: r }),
  limeResult: null,
  setLimeResult: (r) => set({ limeResult: r }),
  shapResult: null,
  setShapResult: (r) => set({ shapResult: r }),
  igResult: null,
  setIgResult: (r) => set({ igResult: r }),
  eli5Result: null,
  setEli5Result: (r) => set({ eli5Result: r }),
  activeXaiMethod: "gradcam",
  setActiveXaiMethod: (m) => set({ activeXaiMethod: m }),
  clearExplainability: () =>
    set({ gradcamResult: null, limeResult: null, shapResult: null, igResult: null, eli5Result: null }),

  scanHistory: [],
  addScanToHistory: (id, result, imagePreview) =>
    set((s) => ({
      scanHistory: [
        { id, timestamp: new Date().toISOString(), result, imagePreview },
        ...s.scanHistory,
      ].slice(0, 50),
    })),
}));
