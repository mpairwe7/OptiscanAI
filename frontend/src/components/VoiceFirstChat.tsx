"use client";

import { useEffect, useRef, useCallback, useMemo } from "react";
import { useVoiceStore, type VoiceMode, type VoiceMessage } from "@/stores/voice-store";
import { useVoiceWebSocket } from "@/hooks/useVoiceWebSocket";

// ── Waveform Canvas ──
// Renders an animated waveform from Float32Array data. In idle/speaking mode
// it shows a gentle ambient wave; in listening mode it mirrors the mic input.
function WaveformCanvas({
  waveformData,
  mode,
}: {
  waveformData: Float32Array;
  mode: VoiceMode;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const phaseRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);

      const w = rect.width;
      const h = rect.height;
      const centerY = h / 2;

      ctx.clearRect(0, 0, w, h);

      // Gradient stroke
      const gradient = ctx.createLinearGradient(0, 0, w, 0);
      gradient.addColorStop(0, "rgba(20, 184, 166, 0.3)");
      gradient.addColorStop(0.5, "rgba(20, 184, 166, 0.9)");
      gradient.addColorStop(1, "rgba(20, 184, 166, 0.3)");

      ctx.strokeStyle = gradient;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";

      const isActive = mode === "listening" || mode === "speaking";
      phaseRef.current += isActive ? 0.06 : 0.02;

      ctx.beginPath();
      const points = waveformData.length || 128;
      const sliceWidth = w / points;

      for (let i = 0; i < points; i++) {
        const x = i * sliceWidth;
        let amplitude: number;

        if (mode === "listening") {
          // Mirror mic input
          amplitude = (waveformData[i] ?? 0) * h * 1.8;
        } else if (mode === "speaking") {
          // Synthetic wave for TTS playback
          const t = phaseRef.current + (i / points) * Math.PI * 4;
          amplitude = Math.sin(t) * h * 0.18 + Math.sin(t * 2.3) * h * 0.08;
        } else if (mode === "processing") {
          // Pulsing sine during processing
          const t = phaseRef.current * 2 + (i / points) * Math.PI * 6;
          amplitude = Math.sin(t) * h * 0.06;
        } else {
          // Idle: subtle ambient wave
          const t = phaseRef.current + (i / points) * Math.PI * 2;
          amplitude = Math.sin(t) * h * 0.04;
        }

        const y = centerY + amplitude;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Secondary wave (phase shifted) for depth
      if (isActive) {
        ctx.globalAlpha = 0.3;
        ctx.beginPath();
        for (let i = 0; i < points; i++) {
          const x = i * sliceWidth;
          let amplitude: number;
          if (mode === "listening") {
            amplitude = (waveformData[Math.max(0, i - 4)] ?? 0) * h * 1.2;
          } else {
            const t = phaseRef.current * 0.8 + (i / points) * Math.PI * 3;
            amplitude = Math.sin(t) * h * 0.12;
          }
          const y = centerY + amplitude;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(rafRef.current);
  }, [waveformData, mode]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-24 sm:h-32"
      aria-hidden="true"
    />
  );
}

// ── Animated Pulse Rings ──
function PulseRings({ mode }: { mode: VoiceMode }) {
  const isActive = mode === "listening" || mode === "speaking";
  const isProcessing = mode === "processing";

  if (!isActive && !isProcessing) return null;

  const ringColor =
    mode === "listening"
      ? "border-teal-400"
      : mode === "speaking"
        ? "border-sky-400"
        : "border-amber-400";

  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none" aria-hidden="true">
      <div
        className={`absolute w-32 h-32 rounded-full ${ringColor} border-2 opacity-30 animate-ping`}
        style={{ animationDuration: "2s" }}
      />
      <div
        className={`absolute w-48 h-48 rounded-full ${ringColor} border opacity-15 animate-ping`}
        style={{ animationDuration: "2.5s", animationDelay: "0.5s" }}
      />
      {isActive && (
        <div
          className={`absolute w-64 h-64 rounded-full ${ringColor} border opacity-10 animate-ping`}
          style={{ animationDuration: "3s", animationDelay: "1s" }}
        />
      )}
    </div>
  );
}

// ── Offline banner ──
function OfflineBanner({ queueCount }: { queueCount: number }) {
  const { lastSyncTime, syncStatus } = useVoiceStore();

  const lastSyncLabel = useMemo(() => {
    if (!lastSyncTime) return "Never synced";
    const diff = Date.now() - lastSyncTime;
    if (diff < 60_000) return "Synced just now";
    if (diff < 3_600_000) return `Synced ${Math.floor(diff / 60_000)}m ago`;
    return `Synced ${Math.floor(diff / 3_600_000)}h ago`;
  }, [lastSyncTime]);

  return (
    <div
      className="flex items-center justify-between px-4 py-2 bg-amber-500/90 text-white text-xs font-medium"
      role="status"
      aria-live="polite"
    >
      <div className="flex items-center gap-2">
        <span className="inline-block w-2 h-2 bg-white rounded-full animate-pulse" />
        <span>Offline Mode</span>
      </div>
      <div className="flex items-center gap-3">
        {queueCount > 0 && (
          <span>
            {queueCount} pending {syncStatus === "syncing" ? "(syncing...)" : ""}
          </span>
        )}
        <span className="opacity-80">{lastSyncLabel}</span>
      </div>
    </div>
  );
}

// ── Status label ──
function StatusLabel({ mode }: { mode: VoiceMode }) {
  const labels: Record<VoiceMode, string> = {
    idle: "Tap to start",
    listening: "Listening...",
    processing: "Thinking...",
    speaking: "Speaking...",
  };

  const colors: Record<VoiceMode, string> = {
    idle: "text-slate-400 dark:text-slate-500",
    listening: "text-teal-500",
    processing: "text-amber-500",
    speaking: "text-sky-500",
  };

  return (
    <p
      className={`text-sm font-medium ${colors[mode]} transition-colors duration-300`}
      aria-live="polite"
    >
      {labels[mode]}
    </p>
  );
}

// ── Message bubble ──
function MessageBubble({ message }: { message: VoiceMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? "bg-teal-600 text-white rounded-br-sm"
            : "bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 rounded-bl-sm"
        }`}
      >
        {message.imageDataUrl && (
          <img
            src={message.imageDataUrl}
            alt="Captured"
            className="w-full max-w-[200px] rounded-lg mb-2"
          />
        )}
        <p>{message.text}</p>
        <time
          className={`block text-[10px] mt-1 ${isUser ? "text-teal-200" : "text-slate-400 dark:text-slate-500"}`}
        >
          {new Date(message.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </time>
      </div>
    </div>
  );
}

// ── Sentence-chunked response display ──
function StreamingResponse({ chunks }: { chunks: string[] }) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Split accumulated text into sentences for chunked display
  const sentences = useMemo(() => {
    const full = chunks.join("");
    if (!full) return [];
    return full.match(/[^.!?]+[.!?]+|[^.!?]+$/g) ?? [full];
  }, [chunks]);

  // Auto-scroll to bottom
  useEffect(() => {
    containerRef.current?.scrollTo({ top: containerRef.current.scrollHeight, behavior: "smooth" });
  }, [sentences.length]);

  if (sentences.length === 0) return null;

  return (
    <div
      ref={containerRef}
      className="bg-slate-100 dark:bg-slate-800 rounded-2xl rounded-bl-sm px-4 py-3 max-w-[85%] text-sm leading-relaxed text-slate-800 dark:text-slate-200 max-h-40 overflow-y-auto"
      role="log"
      aria-label="AI response"
    >
      {sentences.map((sentence, i) => (
        <span
          key={i}
          className="animate-fade-in inline"
          style={{ animationDelay: `${i * 80}ms` }}
        >
          {sentence}
        </span>
      ))}
    </div>
  );
}

// ── Main Component ──
export function VoiceFirstChat() {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const {
    mode,
    waveformData,
    messages,
    responseChunks,
    isOffline: storeOffline,
    offlineQueue,
    settings,
    toggleVoiceVision,
    resetConversation,
  } = useVoiceStore();

  const {
    isConnected,
    isListening,
    isOffline,
    partialTranscript,
    startListening,
    stopListening,
  } = useVoiceWebSocket();

  // Auto-scroll conversation
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, responseChunks.length]);

  // ── Mic button handler ──
  const handleMicPress = useCallback(async () => {
    if (mode === "listening" || isListening) {
      stopListening();
    } else {
      await startListening();
    }
  }, [mode, isListening, startListening, stopListening]);

  // Mic button appearance
  const micButtonClasses = useMemo(() => {
    const base =
      "relative flex items-center justify-center w-20 h-20 sm:w-24 sm:h-24 rounded-full transition-all duration-300 focus:outline-none focus-visible:ring-4 focus-visible:ring-teal-400/50";
    if (mode === "listening") return `${base} bg-teal-500 shadow-lg shadow-teal-500/40 scale-110`;
    if (mode === "processing") return `${base} bg-amber-500 shadow-lg shadow-amber-500/30 cursor-wait`;
    if (mode === "speaking") return `${base} bg-sky-500 shadow-lg shadow-sky-500/30`;
    return `${base} bg-slate-700 dark:bg-slate-600 hover:bg-teal-600 hover:shadow-lg hover:shadow-teal-500/30 active:scale-95`;
  }, [mode]);

  return (
    <div className="flex flex-col h-dvh bg-white dark:bg-slate-950 text-slate-900 dark:text-white select-none">
      {/* ── Offline Banner ── */}
      {(isOffline || storeOffline) && <OfflineBanner queueCount={offlineQueue.length} />}

      {/* ── Header ── */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-slate-200 dark:border-slate-800">
        <div className="flex items-center gap-3">
          <div
            className={`w-2.5 h-2.5 rounded-full ${
              isConnected ? "bg-emerald-500" : isOffline ? "bg-amber-500" : "bg-red-500"
            }`}
            aria-label={isConnected ? "Connected" : isOffline ? "Offline" : "Disconnected"}
          />
          <h1 className="text-base font-semibold">OptiscanAI Voice</h1>
        </div>
        <div className="flex items-center gap-2">
          {/* Voice-Vision toggle */}
          <button
            type="button"
            onClick={toggleVoiceVision}
            className="flex items-center justify-center w-10 h-10 rounded-full text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-400"
            aria-label="Toggle camera mode"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
              <circle cx="12" cy="13" r="4" />
            </svg>
          </button>
          {/* Reset conversation */}
          <button
            type="button"
            onClick={resetConversation}
            className="flex items-center justify-center w-10 h-10 rounded-full text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-400"
            aria-label="Clear conversation"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
              <polyline points="1 4 1 10 7 10" />
              <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
            </svg>
          </button>
        </div>
      </header>

      {/* ── Conversation Feed ── */}
      <div
        className="flex-1 overflow-y-auto px-4 py-4 space-y-1 min-h-0"
        role="log"
        aria-label="Voice conversation"
      >
        {messages.length === 0 && mode === "idle" && (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <div className="w-16 h-16 mb-4 rounded-full bg-teal-50 dark:bg-teal-900/30 flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-8 h-8 text-teal-500">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
            </div>
            <p className="text-sm text-slate-500 dark:text-slate-400 max-w-[260px]">
              Tap the microphone to start a voice conversation about retinal screening
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {/* Live streaming response */}
        {mode === "speaking" && responseChunks.length > 0 && (
          <div className="flex justify-start mb-3">
            <StreamingResponse chunks={responseChunks} />
          </div>
        )}

        {/* Partial transcript (live) */}
        {partialTranscript && mode === "listening" && (
          <div className="flex justify-end mb-3">
            <div className="max-w-[85%] rounded-2xl rounded-br-sm px-4 py-2.5 text-sm bg-teal-600/60 text-white/80 italic">
              {partialTranscript}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* ── Voice Control Area ── */}
      <div className="relative flex flex-col items-center pb-8 pt-2 bg-gradient-to-t from-slate-50 dark:from-slate-900 via-slate-50/80 dark:via-slate-900/80 to-transparent">
        {/* Waveform */}
        <div className="w-full max-w-md px-4">
          <WaveformCanvas waveformData={waveformData} mode={mode} />
        </div>

        {/* Pulse rings container */}
        <div className="relative mt-2">
          <PulseRings mode={mode} />

          {/* Mic button */}
          <button
            type="button"
            onClick={handleMicPress}
            disabled={mode === "processing"}
            className={micButtonClasses}
            aria-label={
              mode === "listening"
                ? "Stop listening"
                : mode === "processing"
                  ? "Processing your request"
                  : mode === "speaking"
                    ? "Tap to interrupt"
                    : "Start voice input"
            }
            aria-pressed={mode === "listening"}
          >
            {/* Mic icon */}
            {mode === "processing" ? (
              <svg className="w-8 h-8 text-white animate-spin" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeDasharray="31.4" strokeDashoffset="10" strokeLinecap="round" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-8 h-8 text-white">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
            )}
          </button>
        </div>

        <StatusLabel mode={mode} />

        {/* Settings info bar */}
        <div className="flex items-center gap-4 mt-2 text-[10px] text-slate-400 dark:text-slate-600">
          <span>Lang: {settings.language}</span>
          <span>VAD: {Math.round(settings.vadSensitivity * 100)}%</span>
          {settings.bargeInEnabled && <span>Barge-in on</span>}
        </div>
      </div>
    </div>
  );
}
