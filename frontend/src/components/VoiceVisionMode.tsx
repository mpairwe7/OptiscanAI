"use client";

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { useVoiceStore, type VoiceMode } from "@/stores/voice-store";
import { useVoiceWebSocket } from "@/hooks/useVoiceWebSocket";

// ── Camera preview ──
function CameraPreview({
  videoRef,
  onCapture,
  isCameraReady,
}: {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  onCapture: () => void;
  isCameraReady: boolean;
}) {
  return (
    <div className="relative w-full h-full bg-black overflow-hidden">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover"
        aria-label="Camera preview"
      />
      {!isCameraReady && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900">
          <div className="flex flex-col items-center gap-3">
            <svg className="w-8 h-8 text-slate-400 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeDasharray="31.4" strokeDashoffset="10" strokeLinecap="round" />
            </svg>
            <p className="text-xs text-slate-400">Starting camera...</p>
          </div>
        </div>
      )}

      {/* Corner frame guides */}
      <div className="absolute inset-4 pointer-events-none" aria-hidden="true">
        <div className="absolute top-0 left-0 w-6 h-6 border-t-2 border-l-2 border-teal-400 rounded-tl" />
        <div className="absolute top-0 right-0 w-6 h-6 border-t-2 border-r-2 border-teal-400 rounded-tr" />
        <div className="absolute bottom-0 left-0 w-6 h-6 border-b-2 border-l-2 border-teal-400 rounded-bl" />
        <div className="absolute bottom-0 right-0 w-6 h-6 border-b-2 border-r-2 border-teal-400 rounded-br" />
      </div>

      {/* Capture button overlay */}
      <div className="absolute bottom-4 left-0 right-0 flex justify-center">
        <button
          type="button"
          onClick={onCapture}
          disabled={!isCameraReady}
          className="flex items-center justify-center w-16 h-16 rounded-full bg-white/20 backdrop-blur-sm border-4 border-white shadow-xl hover:bg-white/30 active:scale-90 transition-all focus:outline-none focus-visible:ring-4 focus-visible:ring-teal-400/50 disabled:opacity-40"
          aria-label="Capture image"
        >
          <div className="w-12 h-12 rounded-full bg-white" />
        </button>
      </div>
    </div>
  );
}

// ── Image preview overlay after capture ──
function CapturePreview({
  imageDataUrl,
  onRetake,
  onSend,
  isSending,
}: {
  imageDataUrl: string;
  onRetake: () => void;
  onSend: () => void;
  isSending: boolean;
}) {
  return (
    <div className="relative w-full h-full bg-black">
      <img
        src={imageDataUrl}
        alt="Captured preview"
        className="w-full h-full object-contain"
      />
      <div className="absolute bottom-4 left-0 right-0 flex items-center justify-center gap-6">
        <button
          type="button"
          onClick={onRetake}
          className="flex items-center justify-center w-14 h-14 rounded-full bg-red-500/80 backdrop-blur-sm text-white hover:bg-red-500 active:scale-90 transition-all focus:outline-none focus-visible:ring-4 focus-visible:ring-red-400/50"
          aria-label="Retake photo"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
        <button
          type="button"
          onClick={onSend}
          disabled={isSending}
          className="flex items-center justify-center w-14 h-14 rounded-full bg-teal-500/80 backdrop-blur-sm text-white hover:bg-teal-500 active:scale-90 transition-all focus:outline-none focus-visible:ring-4 focus-visible:ring-teal-400/50 disabled:opacity-50"
          aria-label="Send image for analysis"
        >
          {isSending ? (
            <svg className="w-6 h-6 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeDasharray="31.4" strokeDashoffset="10" strokeLinecap="round" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}

// ── Voice indicator for bottom panel ──
function VoiceModeIndicator({ mode }: { mode: VoiceMode }) {
  const labels: Record<VoiceMode, string> = {
    idle: "Tap mic to speak",
    listening: "Listening...",
    processing: "Analyzing...",
    speaking: "Response ready",
  };

  const dotColor: Record<VoiceMode, string> = {
    idle: "bg-slate-400",
    listening: "bg-teal-500 animate-pulse",
    processing: "bg-amber-500 animate-pulse",
    speaking: "bg-sky-500 animate-pulse",
  };

  return (
    <div className="flex items-center gap-2">
      <span className={`w-2 h-2 rounded-full ${dotColor[mode]}`} />
      <span className="text-xs font-medium text-slate-600 dark:text-slate-300">
        {labels[mode]}
      </span>
    </div>
  );
}

// ── Main Component ──
export function VoiceVisionMode() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);

  const [isCameraReady, setIsCameraReady] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const {
    mode,
    messages,
    transcript,
    responseChunks,
    isOffline: storeOffline,
    offlineQueue,
    setVoiceVisionState,
  } = useVoiceStore();

  const {
    isConnected,
    isListening,
    isOffline,
    partialTranscript,
    response,
    startListening,
    stopListening,
    sendImageWithTranscript,
  } = useVoiceWebSocket();

  // ── Start camera ──
  useEffect(() => {
    let cancelled = false;

    async function initCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "environment",
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
          audio: false,
        });

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        cameraStreamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setIsCameraReady(true);
        setCameraError(null);
      } catch (err) {
        if (!cancelled) {
          setCameraError(
            err instanceof DOMException && err.name === "NotAllowedError"
              ? "Camera access denied. Please enable camera permissions."
              : "Failed to start camera.",
          );
        }
      }
    }

    initCamera();

    return () => {
      cancelled = true;
      cameraStreamRef.current?.getTracks().forEach((t) => t.stop());
      cameraStreamRef.current = null;
      setIsCameraReady(false);
    };
  }, []);

  // ── Capture photo ──
  const handleCapture = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    setCapturedImage(dataUrl);
  }, []);

  // ── Retake ──
  const handleRetake = useCallback(() => {
    setCapturedImage(null);
  }, []);

  // ── Send image + transcript ──
  const handleSend = useCallback(async () => {
    if (!capturedImage || isSending) return;

    setIsSending(true);
    const currentTranscript = useVoiceStore.getState().transcript;
    const message = currentTranscript || "Analyze this retinal image";

    // Add to conversation immediately (optimistic)
    useVoiceStore.getState().addMessage({
      id: crypto.randomUUID(),
      role: "user",
      text: message,
      timestamp: Date.now(),
      imageDataUrl: capturedImage,
    });

    try {
      await sendImageWithTranscript(capturedImage, message);
    } catch {
      // sendImageWithTranscript handles offline queuing internally
    } finally {
      setCapturedImage(null);
      setIsSending(false);
    }
  }, [capturedImage, isSending, sendImageWithTranscript]);

  // ── Mic toggle ──
  const handleMicPress = useCallback(async () => {
    if (mode === "listening" || isListening) {
      stopListening();
    } else {
      await startListening();
    }
  }, [mode, isListening, startListening, stopListening]);

  // ── Back to voice-only ──
  const handleBack = useCallback(() => {
    cameraStreamRef.current?.getTracks().forEach((t) => t.stop());
    cameraStreamRef.current = null;
    setVoiceVisionState("voice-only");
  }, [setVoiceVisionState]);

  // Mic button styling
  const micBtnClasses = useMemo(() => {
    const base =
      "flex items-center justify-center w-14 h-14 rounded-full transition-all duration-200 focus:outline-none focus-visible:ring-4 focus-visible:ring-teal-400/50";
    if (mode === "listening") return `${base} bg-teal-500 shadow-lg shadow-teal-500/40 scale-105`;
    if (mode === "processing") return `${base} bg-amber-500 cursor-wait`;
    return `${base} bg-slate-700 dark:bg-slate-600 hover:bg-teal-600 active:scale-90`;
  }, [mode]);

  // Latest response for display
  const latestResponse = useMemo(() => {
    if (responseChunks.length > 0) return responseChunks.join("");
    if (response) return response;
    const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
    return lastAssistant?.text ?? null;
  }, [responseChunks, response, messages]);

  return (
    <div className="flex flex-col h-dvh bg-slate-950 text-white select-none">
      {/* Hidden canvas for capture */}
      <canvas ref={canvasRef} className="hidden" aria-hidden="true" />

      {/* ── Offline Banner ── */}
      {(isOffline || storeOffline) && (
        <div
          className="flex items-center justify-between px-4 py-1.5 bg-amber-500/90 text-white text-xs font-medium"
          role="status"
          aria-live="polite"
        >
          <div className="flex items-center gap-2">
            <span className="inline-block w-2 h-2 bg-white rounded-full animate-pulse" />
            <span>Offline - captures stored locally</span>
          </div>
          {offlineQueue.length > 0 && <span>{offlineQueue.length} queued</span>}
        </div>
      )}

      {/* ── Top bar ── */}
      <div className="flex items-center justify-between px-3 py-2 bg-black/60 backdrop-blur-sm z-10">
        <button
          type="button"
          onClick={handleBack}
          className="flex items-center gap-1.5 text-sm text-slate-300 hover:text-white transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-400 rounded-lg px-2 py-1.5"
          aria-label="Back to voice mode"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
            <polyline points="15 18 9 12 15 6" />
          </svg>
          Voice
        </button>

        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${isConnected ? "bg-emerald-500" : "bg-red-500"}`}
            aria-label={isConnected ? "Connected" : "Disconnected"}
          />
          <span className="text-xs text-slate-400">Voice + Vision</span>
        </div>
      </div>

      {/* ── Camera area (top 60%) ── */}
      <div className="relative flex-[3] min-h-0">
        {cameraError ? (
          <div className="flex items-center justify-center h-full bg-slate-900 px-6">
            <div className="text-center">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-10 h-10 mx-auto mb-3 text-red-400">
                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                <line x1="4" y1="4" x2="20" y2="20" />
              </svg>
              <p className="text-sm text-red-300 mb-3">{cameraError}</p>
              <button
                type="button"
                onClick={handleBack}
                className="px-4 py-2 bg-slate-700 rounded-lg text-sm hover:bg-slate-600 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-400"
              >
                Return to Voice
              </button>
            </div>
          </div>
        ) : capturedImage ? (
          <CapturePreview
            imageDataUrl={capturedImage}
            onRetake={handleRetake}
            onSend={handleSend}
            isSending={isSending}
          />
        ) : (
          <CameraPreview
            videoRef={videoRef}
            onCapture={handleCapture}
            isCameraReady={isCameraReady}
          />
        )}
      </div>

      {/* ── Voice controls (bottom 40%) ── */}
      <div className="flex-[2] flex flex-col bg-white dark:bg-slate-900 rounded-t-3xl -mt-4 z-10 min-h-0">
        {/* Voice mode indicator */}
        <div className="flex items-center justify-between px-5 pt-4 pb-2">
          <VoiceModeIndicator mode={mode} />
          {partialTranscript && (
            <span className="text-xs text-teal-500 italic truncate max-w-[50%]">
              {partialTranscript}
            </span>
          )}
        </div>

        {/* Response / transcript area */}
        <div className="flex-1 overflow-y-auto px-5 py-2 min-h-0">
          {transcript && (
            <div className="mb-2">
              <p className="text-[10px] uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-1">
                You said
              </p>
              <p className="text-sm text-slate-700 dark:text-slate-200">{transcript}</p>
            </div>
          )}
          {latestResponse && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-1">
                Analysis
              </p>
              <p className="text-sm text-slate-800 dark:text-slate-100 leading-relaxed">
                {latestResponse}
              </p>
            </div>
          )}
          {!transcript && !latestResponse && (
            <p className="text-sm text-slate-400 dark:text-slate-500 text-center py-4">
              Capture a retinal image or speak a command
            </p>
          )}
        </div>

        {/* Bottom controls */}
        <div className="flex items-center justify-center gap-6 px-5 pb-6 pt-2">
          {/* Capture shortcut (if camera is live and no capture) */}
          {!capturedImage && isCameraReady && (
            <button
              type="button"
              onClick={handleCapture}
              className="flex items-center justify-center w-12 h-12 rounded-full bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600 active:scale-90 transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-400"
              aria-label="Quick capture"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                <circle cx="12" cy="13" r="4" />
              </svg>
            </button>
          )}

          {/* Mic button */}
          <button
            type="button"
            onClick={handleMicPress}
            disabled={mode === "processing"}
            className={micBtnClasses}
            aria-label={
              mode === "listening" ? "Stop listening" : mode === "processing" ? "Processing" : "Start voice input"
            }
            aria-pressed={mode === "listening"}
          >
            {mode === "processing" ? (
              <svg className="w-6 h-6 text-white animate-spin" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeDasharray="31.4" strokeDashoffset="10" strokeLinecap="round" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6 text-white">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
            )}
          </button>

          {/* Settings hint */}
          <button
            type="button"
            onClick={handleBack}
            className="flex items-center justify-center w-12 h-12 rounded-full bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600 active:scale-90 transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-400"
            aria-label="Exit to voice mode"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
