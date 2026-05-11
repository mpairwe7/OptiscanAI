"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import { useVoiceStore } from "@/stores/voice-store";

// ── Constants ──
const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;
const AUDIO_CHUNK_MS = 250;
const AUDIO_MIME = "audio/webm;codecs=opus";

// ── Server event types ──
interface VoiceEvent {
  type:
    | "transcription"
    | "partial_transcription"
    | "response_start"
    | "response_chunk"
    | "response_end"
    | "error"
    | "vad_speech_start"
    | "vad_speech_end";
  data?: string;
  error?: string;
  timestamp?: number;
}

// ── Build the WS URL from the current page origin ──
function buildWsUrl(): string {
  const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";
  const url = new URL(apiBase);
  const protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${url.host}/v1/voice/stream`;
}

export interface UseVoiceWebSocketReturn {
  isConnected: boolean;
  isListening: boolean;
  isOffline: boolean;
  transcript: string;
  partialTranscript: string;
  response: string;
  startListening: () => Promise<void>;
  stopListening: () => void;
  sendImageWithTranscript: (imageDataUrl: string, transcript: string) => void;
}

export function useVoiceWebSocket(): UseVoiceWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const isUnmountedRef = useRef(false);

  const [isConnected, setIsConnected] = useState(false);
  const [isListening, setIsListening] = useState(false);

  const {
    transcript,
    partialTranscript,
    response,
    isOffline,
    settings,
    setMode,
    setTranscript,
    setPartialTranscript,
    setResponse,
    appendResponseChunk,
    clearResponseChunks,
    setWaveformData,
    setOfflineMode,
    addMessage,
  } = useVoiceStore();

  // ── Online/offline detection ──
  useEffect(() => {
    const handleOnline = () => setOfflineMode(false);
    const handleOffline = () => setOfflineMode(true);

    // Set initial state
    setOfflineMode(!navigator.onLine);

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, [setOfflineMode]);

  // ── Handle incoming WS messages ──
  const handleMessage = useCallback(
    (event: MessageEvent) => {
      let parsed: VoiceEvent;
      try {
        parsed = JSON.parse(event.data as string) as VoiceEvent;
      } catch {
        return;
      }

      switch (parsed.type) {
        case "partial_transcription":
          if (parsed.data) setPartialTranscript(parsed.data);
          break;

        case "transcription":
          if (parsed.data) {
            setTranscript(parsed.data);
            addMessage({
              id: crypto.randomUUID(),
              role: "user",
              text: parsed.data,
              timestamp: Date.now(),
            });
          }
          setMode("processing");
          break;

        case "response_start":
          setMode("speaking");
          clearResponseChunks();
          break;

        case "response_chunk":
          if (parsed.data) {
            appendResponseChunk(parsed.data);
            // Build full response from chunks
            const chunks = useVoiceStore.getState().responseChunks;
            setResponse(chunks.join(""));
          }
          break;

        case "response_end": {
          const fullResponse = useVoiceStore.getState().responseChunks.join("");
          setResponse(fullResponse);
          addMessage({
            id: crypto.randomUUID(),
            role: "assistant",
            text: fullResponse,
            timestamp: Date.now(),
          });
          setMode("idle");
          break;
        }

        case "vad_speech_start":
          // Barge-in: user started speaking during TTS
          if (settings.bargeInEnabled) {
            setMode("listening");
            clearResponseChunks();
          }
          break;

        case "vad_speech_end":
          // VAD silence detected
          break;

        case "error":
          console.error("[VoiceWS] Server error:", parsed.error);
          setMode("idle");
          break;
      }
    },
    [
      settings.bargeInEnabled,
      setMode,
      setTranscript,
      setPartialTranscript,
      setResponse,
      appendResponseChunk,
      clearResponseChunks,
      addMessage,
    ],
  );

  // ── WebSocket connection with exponential backoff ──
  const connect = useCallback(() => {
    if (isUnmountedRef.current || isOffline) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const url = buildWsUrl();
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      reconnectAttemptRef.current = 0;

      // Send language preference
      ws.send(JSON.stringify({ type: "config", language: settings.language }));
    };

    ws.onmessage = handleMessage;

    ws.onerror = () => {
      setIsConnected(false);
    };

    ws.onclose = () => {
      setIsConnected(false);
      wsRef.current = null;

      if (isUnmountedRef.current) return;

      // Exponential backoff
      const attempt = reconnectAttemptRef.current;
      const delay = Math.min(RECONNECT_BASE_MS * Math.pow(2, attempt), RECONNECT_MAX_MS);
      reconnectAttemptRef.current = attempt + 1;

      reconnectTimerRef.current = setTimeout(connect, delay);
    };
  }, [isOffline, settings.language, handleMessage]);

  // ── Establish connection on mount ──
  useEffect(() => {
    isUnmountedRef.current = false;
    if (!isOffline) connect();

    return () => {
      isUnmountedRef.current = true;
      clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect, isOffline]);

  // ── Waveform analyser loop ──
  const startWaveformLoop = useCallback(() => {
    if (!analyserRef.current) return;

    const analyser = analyserRef.current;
    const bufferLength = analyser.fftSize;
    const dataArray = new Float32Array(bufferLength);

    let rafId: number;
    const loop = () => {
      if (isUnmountedRef.current) return;
      analyser.getFloatTimeDomainData(dataArray);
      // Downsample to 128 points for the visualization
      const downsampled = new Float32Array(128);
      const step = bufferLength / 128;
      for (let i = 0; i < 128; i++) {
        downsampled[i] = dataArray[Math.floor(i * step)];
      }
      setWaveformData(downsampled);
      rafId = requestAnimationFrame(loop);
    };
    loop();

    return () => cancelAnimationFrame(rafId);
  }, [setWaveformData]);

  // ── Start microphone + streaming ──
  const startListening = useCallback(async () => {
    if (isListening) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        },
      });
      streamRef.current = stream;

      // Set up Web Audio analyser for VAD waveform
      let cleanupWaveform: (() => void) | null | undefined = null;
      try {
        const audioCtx = new AudioContext({ sampleRate: 16000 });
        audioContextRef.current = audioCtx;
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        analyserRef.current = analyser;
        cleanupWaveform = startWaveformLoop();
      } catch (audioErr) {
        console.warn("[VoiceWS] AudioContext unavailable, continuing without waveform:", audioErr);
      }

      // MediaRecorder for streaming chunks
      try {
        if (MediaRecorder.isTypeSupported(AUDIO_MIME)) {
          const recorder = new MediaRecorder(stream, {
            mimeType: AUDIO_MIME,
            audioBitsPerSecond: 32000,
          });
          mediaRecorderRef.current = recorder;

          recorder.ondataavailable = (e) => {
            if (e.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(e.data);
            }
          };

          recorder.onstop = () => {
            cleanupWaveform?.();
          };

          recorder.onerror = () => {
            cleanupWaveform?.();
          };

          recorder.start(AUDIO_CHUNK_MS);
        }
      } catch (recErr) {
        console.warn("[VoiceWS] MediaRecorder failed, microphone may not stream:", recErr);
        cleanupWaveform?.();
      }

      setIsListening(true);
      setMode("listening");
    } catch (err) {
      console.error("[VoiceWS] Microphone access denied:", err);
      setMode("idle");
    }
  }, [isListening, setMode, startWaveformLoop]);

  // ── Stop microphone ──
  const stopListening = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    mediaRecorderRef.current = null;

    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    audioContextRef.current?.close().catch(() => {});
    audioContextRef.current = null;
    analyserRef.current = null;

    setIsListening(false);

    // Tell server that audio ended
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "audio_end" }));
    }

    setMode("processing");
  }, [setMode]);

  // ── Send image + transcript (for voice-vision mode) ──
  const sendImageWithTranscript = useCallback(
    (imageDataUrl: string, currentTranscript: string) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: "image_with_transcript",
            image: imageDataUrl,
            transcript: currentTranscript,
          }),
        );
        setMode("processing");
      } else if (isOffline) {
        // Queue for offline sync
        const { enqueueOfflineCapture } = useVoiceStore.getState();
        enqueueOfflineCapture({
          id: crypto.randomUUID(),
          transcript: currentTranscript,
          imageDataUrl,
          createdAt: Date.now(),
        });
      }
    },
    [isOffline, setMode],
  );

  return {
    isConnected,
    isListening,
    isOffline,
    transcript,
    partialTranscript,
    response,
    startListening,
    stopListening,
    sendImageWithTranscript,
  };
}
