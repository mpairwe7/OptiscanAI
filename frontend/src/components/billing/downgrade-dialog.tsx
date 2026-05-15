"use client";
import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { apiChangePlan } from "@/lib/auth-api";
import { ApiError } from "@/lib/api-fetch";

interface Props {
  currentPlanName: string;
  currentProvider: string;
  onClose: () => void;
}

export function DowngradeDialog({ currentPlanName, currentProvider, onClose }: Props) {
  const qc = useQueryClient();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  async function confirm() {
    setSubmitting(true);
    setError(null);
    try {
      await apiChangePlan("free", "monthly");
      qc.invalidateQueries({ queryKey: ["auth", "me"] });
      qc.invalidateQueries({ queryKey: ["billing"] });
      qc.invalidateQueries({ queryKey: ["orgs"] });
      setDone(true);
    } catch (err) {
      // Server returns detail as either string or object
      let msg = "Downgrade failed";
      if (err instanceof ApiError) {
        const body = err.body as { detail?: unknown } | null;
        if (body && typeof body.detail === "object" && body.detail !== null && "message" in body.detail) {
          msg = String((body.detail as { message: string }).message);
        } else {
          msg = err.message;
        }
      }
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  }

  if (done) {
    return (
      <div
        className="fixed inset-0 z-50 flex items-center justify-center p-4 mobile-overlay"
        role="dialog"
        aria-modal="true"
      >
        <div className="w-full max-w-md bg-white rounded-2xl shadow-2xl p-6 text-center animate-slide-up">
          <div className="w-12 h-12 mx-auto rounded-full bg-teal-50 text-teal-600 flex items-center justify-center text-2xl">
            ✓
          </div>
          <h2 className="mt-4 text-xl font-bold text-slate-900">
            Downgraded to Free
          </h2>
          <p className="mt-2 text-sm text-slate-600">
            Your account is on the Free plan effective now. Stripe will issue a prorated credit for any
            unused time on {currentPlanName}.
          </p>
          <button
            onClick={onClose}
            className="mt-5 inline-flex items-center px-4 py-2 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white"
          >
            OK
          </button>
        </div>
      </div>
    );
  }

  const isStripe = currentProvider === "stripe";
  const isMoMo = currentProvider === "mtn" || currentProvider === "airtel" || currentProvider === "flutterwave";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 mobile-overlay"
      role="dialog"
      aria-modal="true"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md bg-white rounded-2xl shadow-2xl p-6 animate-slide-up"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="w-10 h-10 rounded-full bg-amber-50 text-amber-600 flex items-center justify-center">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
          </svg>
        </div>
        <h2 className="mt-3 text-xl font-bold text-slate-900">
          Downgrade {currentPlanName} → Free?
        </h2>
        <p className="mt-2 text-sm text-slate-600">
          Your account moves to the Free plan <span className="font-semibold">immediately</span>.
          You&apos;ll keep access to 10 scans / month, Grad-CAM only, and a single seat.
        </p>

        <ul className="mt-4 space-y-1.5 text-sm text-slate-700">
          {isStripe && (
            <li className="flex items-start gap-2">
              <span className="text-teal-600 mt-0.5">✓</span>
              <span>Stripe will cancel the subscription now and issue a prorated credit for the unused time.</span>
            </li>
          )}
          {isMoMo && (
            <li className="flex items-start gap-2">
              <span className="text-amber-600 mt-0.5">!</span>
              <span>You already paid for the current period — that time is forfeit (mobile money doesn&apos;t prorate refunds).</span>
            </li>
          )}
          <li className="flex items-start gap-2">
            <span className="text-amber-600 mt-0.5">!</span>
            <span>SHAP, LIME, Integrated Gradients, ELI5, clinical reasoning, voice mode, and team features lock immediately.</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 mt-0.5">!</span>
            <span>Reports older than 7 days will be hidden (not deleted — re-upgrade to restore visibility).</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 mt-0.5">!</span>
            <span>You must remove additional members first if your team has more than 1 active seat.</span>
          </li>
        </ul>

        {error && (
          <div className="mt-4 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">
            {error}
          </div>
        )}

        <div className="mt-6 flex gap-2 justify-end">
          <button
            onClick={onClose}
            disabled={submitting}
            className="inline-flex items-center px-4 py-2 text-sm font-medium rounded-lg border border-slate-300 hover:bg-slate-50 text-slate-700"
          >
            Keep {currentPlanName}
          </button>
          <button
            onClick={confirm}
            disabled={submitting}
            className="inline-flex items-center px-4 py-2 text-sm font-semibold rounded-lg bg-red-600 hover:bg-red-700 text-white disabled:opacity-50"
          >
            {submitting ? "Downgrading…" : "Yes, downgrade to Free"}
          </button>
        </div>
      </div>
    </div>
  );
}
