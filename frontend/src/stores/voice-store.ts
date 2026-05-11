import { create } from "zustand";
import { persist } from "zustand/middleware";

// ── Voice mode states ──
export type VoiceMode = "idle" | "listening" | "processing" | "speaking";

// ── Sync status for offline queue ──
export type SyncStatus = "idle" | "syncing" | "error" | "complete";

// ── Voice+Vision combined mode ──
export type VoiceVisionState = "voice-only" | "voice-vision";

// ── Conversation message ──
export interface VoiceMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: number;
  imageDataUrl?: string;
}

// ── Offline capture (queued for sync) ──
export interface OfflineCapture {
  id: string;
  transcript: string;
  imageDataUrl?: string;
  createdAt: number;
}

interface VoiceSettings {
  language: string;
  speechRate: number;
  voiceId: string;
  vadSensitivity: number;
  bargeInEnabled: boolean;
}

interface VoiceState {
  // Voice state
  mode: VoiceMode;
  transcript: string;
  partialTranscript: string;
  response: string;
  responseChunks: string[];
  waveformData: Float32Array;
  voiceVisionState: VoiceVisionState;

  // Conversation history
  messages: VoiceMessage[];

  // Offline state
  isOffline: boolean;
  lastSyncTime: number | null;
  syncStatus: SyncStatus;
  offlineBundleVersion: string | null;
  offlineBundleSize: number | null;
  offlineQueue: OfflineCapture[];

  // Voice settings (persisted)
  settings: VoiceSettings;

  // Actions — voice
  startVoice: () => void;
  stopVoice: () => void;
  setMode: (mode: VoiceMode) => void;
  setTranscript: (text: string) => void;
  setPartialTranscript: (text: string) => void;
  setResponse: (text: string) => void;
  appendResponseChunk: (chunk: string) => void;
  clearResponseChunks: () => void;
  setWaveformData: (data: Float32Array) => void;
  addMessage: (msg: VoiceMessage) => void;
  clearMessages: () => void;

  // Actions — offline
  setOfflineMode: (offline: boolean) => void;
  updateSyncStatus: (status: SyncStatus) => void;
  setLastSyncTime: (time: number) => void;
  setOfflineBundle: (version: string, size: number) => void;
  enqueueOfflineCapture: (capture: OfflineCapture) => void;
  dequeueOfflineCapture: (id: string) => void;
  clearOfflineQueue: () => void;

  // Actions — voice-vision toggle
  toggleVoiceVision: () => void;
  setVoiceVisionState: (state: VoiceVisionState) => void;

  // Actions — settings
  updateSettings: (patch: Partial<VoiceSettings>) => void;
  resetConversation: () => void;
}

const DEFAULT_SETTINGS: VoiceSettings = {
  language: "en-US",
  speechRate: 1.0,
  voiceId: "default",
  vadSensitivity: 0.6,
  bargeInEnabled: true,
};

export const useVoiceStore = create<VoiceState>()(
  persist(
    (set) => ({
      // Voice state defaults
      mode: "idle",
      transcript: "",
      partialTranscript: "",
      response: "",
      responseChunks: [],
      waveformData: new Float32Array(128),
      voiceVisionState: "voice-only",

      // Conversation
      messages: [],

      // Offline defaults
      isOffline: false,
      lastSyncTime: null,
      syncStatus: "idle",
      offlineBundleVersion: null,
      offlineBundleSize: null,
      offlineQueue: [],

      // Settings
      settings: DEFAULT_SETTINGS,

      // ── Voice actions ──
      startVoice: () =>
        set({ mode: "listening", transcript: "", partialTranscript: "", response: "", responseChunks: [] }),

      stopVoice: () =>
        set({ mode: "idle", partialTranscript: "" }),

      setMode: (mode) => set({ mode }),

      setTranscript: (text) => set({ transcript: text, partialTranscript: "" }),

      setPartialTranscript: (text) => set({ partialTranscript: text }),

      setResponse: (text) => set({ response: text }),

      appendResponseChunk: (chunk) =>
        set((s) => ({ responseChunks: [...s.responseChunks, chunk] })),

      clearResponseChunks: () => set({ responseChunks: [], response: "" }),

      setWaveformData: (data) => set({ waveformData: new Float32Array(data) }),

      addMessage: (msg) =>
        set((s) => ({ messages: [...s.messages, msg].slice(-100) })),

      clearMessages: () => set({ messages: [] }),

      // ── Offline actions ──
      setOfflineMode: (offline) => set({ isOffline: offline }),

      updateSyncStatus: (status) => set({ syncStatus: status }),

      setLastSyncTime: (time) => set({ lastSyncTime: time }),

      setOfflineBundle: (version, size) =>
        set({ offlineBundleVersion: version, offlineBundleSize: size }),

      enqueueOfflineCapture: (capture) =>
        set((s) => ({ offlineQueue: [...s.offlineQueue, capture] })),

      dequeueOfflineCapture: (id) =>
        set((s) => ({ offlineQueue: s.offlineQueue.filter((c) => c.id !== id) })),

      clearOfflineQueue: () => set({ offlineQueue: [] }),

      // ── Voice-vision ──
      toggleVoiceVision: () =>
        set((s) => ({
          voiceVisionState: s.voiceVisionState === "voice-only" ? "voice-vision" : "voice-only",
        })),

      setVoiceVisionState: (state) => set({ voiceVisionState: state }),

      // ── Settings ──
      updateSettings: (patch) =>
        set((s) => ({ settings: { ...s.settings, ...patch } })),

      resetConversation: () =>
        set({
          mode: "idle",
          transcript: "",
          partialTranscript: "",
          response: "",
          responseChunks: [],
          messages: [],
        }),
    }),
    {
      name: "retinal-voice-settings",
      // Only persist settings and offline metadata, not ephemeral voice state
      partialize: (state) => ({
        settings: state.settings,
        lastSyncTime: state.lastSyncTime,
        offlineBundleVersion: state.offlineBundleVersion,
        offlineBundleSize: state.offlineBundleSize,
        offlineQueue: state.offlineQueue,
      }),
    },
  ),
);
