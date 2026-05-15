"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useQueryClient } from "@tanstack/react-query";
import { apiMe } from "@/lib/auth-api";

export default function CheckoutSuccess() {
  const qc = useQueryClient();
  // After redirect from Stripe/Flutterwave, the webhook may not have hit yet.
  // Poll /me every 2s for up to 30s waiting for plan to be paid.
  const [confirmed, setConfirmed] = useState<boolean | null>(null);
  const [planName, setPlanName] = useState<string | null>(null);

  useEffect(() => {
    let attempts = 0;
    let cancelled = false;

    async function tick() {
      attempts += 1;
      try {
        const me = await apiMe();
        const code = me.subscription?.plan.code ?? "free";
        if (code !== "free") {
          if (!cancelled) {
            setConfirmed(true);
            setPlanName(me.subscription?.plan.display_name ?? code);
          }
          qc.invalidateQueries({ queryKey: ["auth", "me"] });
          qc.invalidateQueries({ queryKey: ["billing"] });
          return;
        }
      } catch {
        // ignore transient errors
      }
      if (attempts >= 15) {
        if (!cancelled) setConfirmed(false);
        return;
      }
      setTimeout(() => {
        if (!cancelled) tick();
      }, 2000);
    }
    tick();
    return () => {
      cancelled = true;
    };
  }, [qc]);

  if (confirmed === null) {
    return (
      <div className="max-w-md mx-auto text-center rounded-2xl border border-slate-200 bg-white p-8">
        <div className="w-12 h-12 mx-auto skeleton rounded-full" />
        <h1 className="mt-4 text-xl font-bold text-slate-900">Confirming your payment…</h1>
        <p className="mt-2 text-sm text-slate-600">
          This usually takes a few seconds. Please don&apos;t close this tab.
        </p>
      </div>
    );
  }

  if (confirmed === false) {
    return (
      <div className="max-w-md mx-auto text-center rounded-2xl border border-slate-200 bg-white p-8">
        <div className="w-12 h-12 mx-auto rounded-full bg-amber-50 text-amber-600 flex items-center justify-center text-2xl">
          ⏳
        </div>
        <h1 className="mt-4 text-xl font-bold text-slate-900">Payment processing</h1>
        <p className="mt-2 text-sm text-slate-600">
          Your payment is being verified. Your plan will update automatically — refresh
          /app/billing in a minute, or contact{" "}
          <a href="mailto:support@makstartup.com" className="underline">
            support
          </a>{" "}
          if you don&apos;t see it soon.
        </p>
        <Link
          href="/app/billing"
          className="mt-6 inline-flex items-center justify-center px-4 py-2.5 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white"
        >
          Go to billing
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-md mx-auto text-center rounded-2xl border border-slate-200 bg-white p-8">
      <div className="w-14 h-14 mx-auto rounded-full bg-teal-50 text-teal-600 flex items-center justify-center text-2xl">
        ✓
      </div>
      <h1 className="mt-4 text-2xl font-bold text-slate-900">
        Welcome to {planName ?? "OptiscanAI"}
      </h1>
      <p className="mt-2 text-sm text-slate-600">Your subscription is active.</p>
      <Link
        href="/app/screening"
        className="mt-6 inline-flex items-center justify-center px-4 py-2.5 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white"
      >
        Start screening
      </Link>
    </div>
  );
}
