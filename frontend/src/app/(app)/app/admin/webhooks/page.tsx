"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  apiGetWebhookEvent,
  apiListWebhookEvents,
  apiMe,
  apiReplayWebhookEvent,
  type WebhookEventListItem,
} from "@/lib/auth-api";
import { ApiError } from "@/lib/api-fetch";

const PROVIDERS = ["all", "stripe", "mtn", "airtel", "flutterwave"] as const;
const STATES = [
  { id: "all", label: "All" },
  { id: "error", label: "Errors" },
  { id: "pending", label: "Pending" },
  { id: "ok", label: "OK" },
] as const;

type ProviderId = (typeof PROVIDERS)[number];
type StateId = (typeof STATES)[number]["id"];

function statusOf(e: WebhookEventListItem): "ok" | "error" | "pending" {
  if (e.error) return "error";
  if (!e.processed_at) return "pending";
  return "ok";
}

function relTime(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export default function WebhooksAdminPage() {
  const me = useQuery({ queryKey: ["auth", "me"], queryFn: apiMe });
  const [provider, setProvider] = useState<ProviderId>("all");
  const [state, setState] = useState<StateId>("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const events = useQuery({
    queryKey: ["admin", "webhook-events", provider, state],
    queryFn: () =>
      apiListWebhookEvents({
        provider: provider === "all" ? undefined : provider,
        state: state === "all" ? undefined : state,
        limit: 200,
      }),
    refetchInterval: 15_000,
  });

  if (me.isLoading) return <div className="skeleton h-32 rounded-2xl" />;
  if (!me.data) return null;
  if (!me.data.is_superuser) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-6 max-w-md">
        <h1 className="text-xl font-bold text-slate-900">Restricted</h1>
        <p className="mt-1.5 text-sm text-slate-600">This page is available to platform superusers only.</p>
        <Link
          href="/app/dashboard"
          className="mt-4 inline-flex items-center px-4 py-2 text-sm font-semibold rounded-lg border border-slate-300 hover:bg-slate-50 text-slate-700"
        >
          Back to dashboard
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <header className="flex items-baseline justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Webhook events</h1>
          <p className="mt-1 text-sm text-slate-600">
            Recent provider webhooks. Auto-refreshes every 15 s.
          </p>
        </div>
        <div className="text-xs text-slate-500 font-mono">
          {events.data?.length ?? 0} rows
          {events.isFetching && <span className="ml-2 text-teal-600">syncing…</span>}
        </div>
      </header>

      <div className="flex items-center gap-2 flex-wrap">
        <div className="inline-flex rounded-lg bg-slate-100 p-1 text-xs font-semibold">
          {PROVIDERS.map((p) => (
            <button
              key={p}
              onClick={() => setProvider(p)}
              className={`px-3 py-1.5 rounded-md transition-colors ${
                provider === p ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              {p}
            </button>
          ))}
        </div>
        <div className="inline-flex rounded-lg bg-slate-100 p-1 text-xs font-semibold">
          {STATES.map((s) => (
            <button
              key={s.id}
              onClick={() => setState(s.id)}
              className={`px-3 py-1.5 rounded-md transition-colors ${
                state === s.id ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <section className="rounded-2xl border border-slate-200 bg-white overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-slate-500 text-xs uppercase tracking-wider">
            <tr>
              <th className="py-2.5 px-4 text-left font-medium">Status</th>
              <th className="py-2.5 px-4 text-left font-medium">Provider</th>
              <th className="py-2.5 px-4 text-left font-medium">Event type</th>
              <th className="py-2.5 px-4 text-left font-medium">Received</th>
              <th className="py-2.5 px-4 text-left font-medium">Reference</th>
              <th className="py-2.5 px-4 text-right font-medium" />
            </tr>
          </thead>
          <tbody>
            {events.isLoading && (
              <tr><td colSpan={6} className="py-8 text-center text-slate-500">Loading…</td></tr>
            )}
            {events.data?.length === 0 && !events.isLoading && (
              <tr><td colSpan={6} className="py-8 text-center text-slate-500">No webhook events match these filters.</td></tr>
            )}
            {events.data?.map((e) => {
              const s = statusOf(e);
              return (
                <tr
                  key={e.id}
                  className="border-t border-slate-100 hover:bg-slate-50/60 cursor-pointer"
                  onClick={() => setSelectedId(e.id)}
                >
                  <td className="py-2.5 px-4">
                    {s === "ok" && (
                      <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-700">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" /> ok
                      </span>
                    )}
                    {s === "error" && (
                      <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-red-700">
                        <span className="w-1.5 h-1.5 rounded-full bg-red-500" /> error
                      </span>
                    )}
                    {s === "pending" && (
                      <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-amber-700">
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse-dot" /> pending
                      </span>
                    )}
                  </td>
                  <td className="py-2.5 px-4 capitalize font-medium text-slate-900">{e.provider}</td>
                  <td className="py-2.5 px-4 font-mono text-xs text-slate-700">{e.event_type ?? "—"}</td>
                  <td className="py-2.5 px-4 text-slate-600" title={new Date(e.received_at).toISOString()}>
                    {relTime(e.received_at)}
                  </td>
                  <td className="py-2.5 px-4 font-mono text-xs text-slate-500 truncate max-w-[200px]">
                    {e.provider_event_id}
                  </td>
                  <td className="py-2.5 px-4 text-right">
                    <span className="text-xs text-teal-700">View →</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      {selectedId && (
        <EventDrawer
          eventId={selectedId}
          onClose={() => setSelectedId(null)}
          onReplayed={() => events.refetch()}
        />
      )}
    </div>
  );
}

function EventDrawer({
  eventId,
  onClose,
  onReplayed,
}: {
  eventId: string;
  onClose: () => void;
  onReplayed: () => void;
}) {
  const qc = useQueryClient();
  const detail = useQuery({
    queryKey: ["admin", "webhook-event", eventId],
    queryFn: () => apiGetWebhookEvent(eventId),
  });

  const [replaying, setReplaying] = useState(false);
  const [replayResult, setReplayResult] = useState<unknown | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function handleReplay() {
    setReplaying(true);
    setError(null);
    setReplayResult(null);
    try {
      const res = await apiReplayWebhookEvent(eventId);
      setReplayResult(res);
      qc.invalidateQueries({ queryKey: ["admin", "webhook-events"] });
      qc.invalidateQueries({ queryKey: ["admin", "webhook-event", eventId] });
      qc.invalidateQueries({ queryKey: ["billing"] });
      onReplayed();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Replay failed");
    } finally {
      setReplaying(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-stretch justify-end mobile-overlay"
      role="dialog"
      aria-modal="true"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-2xl bg-white shadow-2xl flex flex-col animate-slide-in"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="px-6 py-4 border-b border-slate-200 flex items-start justify-between gap-3">
          <div>
            <div className="text-xs uppercase tracking-wider font-bold text-slate-500">
              Webhook event
            </div>
            <h2 className="mt-0.5 font-mono text-sm text-slate-900 break-all">
              {detail.data?.provider_event_id ?? eventId}
            </h2>
            {detail.data && (
              <div className="mt-1.5 text-xs text-slate-500">
                <span className="font-mono">{detail.data.event_type}</span> ·{" "}
                <span className="capitalize">{detail.data.provider}</span> ·{" "}
                received {relTime(detail.data.received_at)}
              </div>
            )}
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="text-slate-400 hover:text-slate-700 w-8 h-8 rounded-lg hover:bg-slate-100"
          >
            ✕
          </button>
        </header>

        <div className="px-6 py-4 border-b border-slate-100 flex items-center gap-2 flex-wrap">
          <button
            onClick={handleReplay}
            disabled={replaying || !detail.data?.payload}
            className="inline-flex items-center px-4 py-1.5 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white disabled:opacity-50"
          >
            {replaying ? "Replaying…" : "Replay event"}
          </button>
          {!detail.data?.payload && detail.data !== undefined && (
            <span className="text-xs text-slate-500">
              No payload stored — pre-replay-feature row, cannot replay.
            </span>
          )}
          {detail.data?.error && (
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-red-700 bg-red-50 px-2 py-0.5 rounded">
              last error: {detail.data.error.slice(0, 80)}
            </span>
          )}
        </div>

        {error && (
          <div className="mx-6 mt-4 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">
            {error}
          </div>
        )}

        {replayResult !== null && (
          <div className="mx-6 mt-4 rounded-lg bg-emerald-50 border border-emerald-200 px-3 py-2 text-xs">
            <div className="font-semibold text-emerald-900">Replay result</div>
            <pre className="mt-1.5 whitespace-pre-wrap font-mono text-emerald-900 break-all">
              {JSON.stringify(replayResult, null, 2)}
            </pre>
          </div>
        )}

        <div className="flex-1 overflow-auto p-6 bg-slate-50">
          <div className="text-xs uppercase tracking-wider font-bold text-slate-500 mb-2">
            Raw payload
          </div>
          {detail.isLoading && <div className="skeleton h-40 rounded-lg" />}
          {detail.data?.payload ? (
            <pre className="text-xs font-mono bg-white border border-slate-200 rounded-lg p-4 overflow-x-auto whitespace-pre-wrap break-all max-h-[60vh] overflow-y-auto">
              {JSON.stringify(detail.data.payload, null, 2)}
            </pre>
          ) : (
            !detail.isLoading && (
              <p className="text-sm text-slate-500">Payload not stored.</p>
            )
          )}
        </div>
      </div>
    </div>
  );
}
