"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiGetSeats, apiUpdateSeats, type SeatStateDTO } from "@/lib/auth-api";
import { ApiError } from "@/lib/api-fetch";

function fmtUsd(cents: number) {
  return `$${(cents / 100).toFixed(0)}`;
}

export function SeatManager() {
  const qc = useQueryClient();
  const seats = useQuery({ queryKey: ["billing", "seats"], queryFn: apiGetSeats, retry: 0 });
  const [target, setTarget] = useState<number | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    if (seats.data && target === null) {
      setTarget(seats.data.additional_seats);
    }
  }, [seats.data, target]);

  if (seats.isLoading) return <div className="skeleton h-32 rounded-2xl" />;
  if (!seats.data) return null;
  const s: SeatStateDTO = seats.data;
  const current = target ?? s.additional_seats;
  const totalEffective = s.included_seats + current;
  const delta = current - s.additional_seats;
  const monthlyCost = current * s.per_seat_cents;
  const dirty = current !== s.additional_seats;

  if (!s.can_buy_more) {
    if (s.effective_limit === null) {
      return (
        <section className="rounded-2xl border border-slate-200 bg-white p-6">
          <h2 className="font-semibold text-slate-900">Seats</h2>
          <p className="mt-1.5 text-sm text-slate-600">
            Your Health System plan includes <span className="font-mono font-semibold">unlimited</span> seats.
          </p>
        </section>
      );
    }
    return (
      <section className="rounded-2xl border border-slate-200 bg-white p-6">
        <h2 className="font-semibold text-slate-900">Seats</h2>
        <p className="mt-1.5 text-sm text-slate-600">
          You have <span className="font-mono font-semibold">{s.seats_used} / {s.effective_limit}</span> seats in use.
        </p>
        <p className="mt-3 text-xs text-slate-500">
          Extra-seat purchases are only available on Practice with a Stripe card subscription.{" "}
          <Link href="/contact-sales" className="text-teal-700 underline">
            Email sales
          </Link>{" "}
          if you need to add seats on a mobile-money plan.
        </p>
      </section>
    );
  }

  async function commit() {
    if (!dirty) return;
    setSubmitting(true);
    setError(null);
    try {
      await apiUpdateSeats(current);
      qc.invalidateQueries({ queryKey: ["billing"] });
      qc.invalidateQueries({ queryKey: ["orgs"] });
      qc.invalidateQueries({ queryKey: ["auth", "me"] });
      setConfirming(false);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? typeof err.message === "string"
            ? err.message
            : "Seat update failed"
          : "Seat update failed",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-6">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <h2 className="font-semibold text-slate-900">Seats</h2>
        <div className="text-sm text-slate-500">
          <span className="font-mono font-semibold text-slate-900">{s.seats_used}</span> in use ·{" "}
          <span className="font-mono font-semibold text-slate-900">
            {s.included_seats} included
          </span>{" "}
          ·{" "}
          <span className="font-mono font-semibold text-slate-900">
            {s.additional_seats} extra
          </span>
        </div>
      </header>

      <div className="mt-5 grid gap-5 sm:grid-cols-[1fr_auto] items-center">
        <div>
          <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider">
            Extra seats
          </label>
          <div className="mt-2 flex items-center gap-3">
            <button
              type="button"
              onClick={() => setTarget(Math.max(0, current - 1))}
              disabled={current <= 0}
              className="w-9 h-9 rounded-lg border border-slate-300 hover:bg-slate-50 disabled:opacity-50 text-xl"
              aria-label="Remove one seat"
            >
              −
            </button>
            <input
              type="number"
              min={0}
              max={500}
              value={current}
              onChange={(e) => setTarget(Math.max(0, Number(e.target.value || 0)))}
              className="w-20 px-3 py-1.5 text-center text-xl font-mono font-semibold rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            />
            <button
              type="button"
              onClick={() => setTarget(Math.min(500, current + 1))}
              className="w-9 h-9 rounded-lg border border-slate-300 hover:bg-slate-50 text-xl"
              aria-label="Add one seat"
            >
              +
            </button>
            <span className="text-sm text-slate-500">
              ×{" "}
              <span className="font-mono">
                {fmtUsd(s.per_seat_cents)}
                {s.cycle === "annual" ? "/yr" : "/mo"}
              </span>{" "}
              each
            </span>
          </div>
        </div>

        <div className="text-right">
          <div className="text-xs uppercase tracking-wider font-semibold text-slate-500">
            New effective limit
          </div>
          <div className="mt-1 text-2xl font-bold text-slate-900 font-mono">{totalEffective} seats</div>
          {dirty && (
            <div className="mt-0.5 text-xs text-slate-500">
              Add-on:{" "}
              <span className="font-mono font-semibold text-slate-900">{fmtUsd(monthlyCost)}</span>
              {s.cycle === "annual" ? "/yr" : "/mo"}
            </div>
          )}
        </div>
      </div>

      {totalEffective < s.seats_used && (
        <div className="mt-4 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">
          You have {s.seats_used} active members. Remove members first, then lower the seat count.
        </div>
      )}

      {error && (
        <div className="mt-4 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="mt-5 flex items-center gap-3 flex-wrap">
        {dirty && !confirming && (
          <button
            onClick={() => setConfirming(true)}
            disabled={totalEffective < s.seats_used}
            className="inline-flex items-center px-4 py-2 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white disabled:opacity-50"
          >
            {delta > 0 ? `Add ${delta} seat${delta === 1 ? "" : "s"}` : `Remove ${-delta} seat${-delta === 1 ? "" : "s"}`}
          </button>
        )}
        {confirming && (
          <>
            <button
              onClick={commit}
              disabled={submitting}
              className="inline-flex items-center px-4 py-2 text-sm font-semibold rounded-lg bg-slate-900 hover:bg-slate-800 text-white disabled:opacity-50"
            >
              {submitting
                ? "Updating Stripe…"
                : delta > 0
                  ? `Charge ${fmtUsd(monthlyCost - s.additional_seats * s.per_seat_cents)}${s.cycle === "annual" ? "/yr" : "/mo"} more`
                  : `Confirm reduction`}
            </button>
            <button
              onClick={() => setConfirming(false)}
              disabled={submitting}
              className="inline-flex items-center px-4 py-2 text-sm font-medium rounded-lg border border-slate-300 hover:bg-slate-50 text-slate-700"
            >
              Cancel
            </button>
          </>
        )}
        {dirty && (
          <button
            onClick={() => {
              setTarget(s.additional_seats);
              setConfirming(false);
            }}
            disabled={submitting}
            className="text-sm text-slate-500 hover:text-slate-900"
          >
            Reset
          </button>
        )}
      </div>

      <p className="mt-4 text-xs text-slate-500">
        Stripe pro-rates the change immediately — adding seats charges the prorated amount today,
        removing seats credits your next invoice.
      </p>
    </section>
  );
}
