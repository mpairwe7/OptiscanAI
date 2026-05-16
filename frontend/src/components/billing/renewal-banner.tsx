"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { apiGetSubscription } from "@/lib/auth-api";

function daysUntil(iso: string): number {
  const ms = new Date(iso).getTime() - Date.now();
  return Math.ceil(ms / 86_400_000);
}

interface BannerProps {
  /** "inline" lives on /app/billing as a card; "global" is the slim strip at the top of the AppShell. */
  variant: "global" | "inline";
}

export function RenewalBanner({ variant }: BannerProps) {
  const sub = useQuery({ queryKey: ["billing", "subscription"], queryFn: apiGetSubscription, retry: 0 });
  const [dismissedKey, setDismissedKey] = useState<string | null>(null);

  // Hydrate dismissed state once the query resolves
  useEffect(() => {
    if (variant !== "global" || !sub.data) return;
    const key = `renewal-dismissed:${sub.data.current_period_end}`;
    if (typeof window !== "undefined" && window.localStorage.getItem(key)) {
      setDismissedKey(key);
    } else {
      setDismissedKey(null);
    }
  }, [variant, sub.data]);

  const visibility = useMemo(() => {
    if (!sub.data) return null;
    if (sub.data.provider !== "mtn") return null;
    if (sub.data.cancel_at_period_end) return null;
    if (sub.data.plan_code === "free") return null;
    const days = daysUntil(sub.data.current_period_end);
    if (days > 7) return null;
    return {
      days,
      providerLabel: "MTN MoMo",
      planCode: sub.data.plan_code,
      planName: sub.data.plan_display_name,
      billingCycle: sub.data.billing_cycle as "monthly" | "annual",
      periodEnd: sub.data.current_period_end,
    };
  }, [sub.data]);

  if (!visibility) return null;
  if (variant === "global" && dismissedKey) return null;

  const urgency =
    visibility.days <= 0
      ? "expired"
      : visibility.days === 1
        ? "tomorrow"
        : `in ${visibility.days} days`;

  const checkoutHref = `/app/checkout/${visibility.planCode}?cycle=${visibility.billingCycle}`;
  const tone =
    visibility.days <= 1
      ? { bg: "bg-red-50", border: "border-red-300", text: "text-red-900", chipBg: "bg-red-100", chipText: "text-red-700" }
      : visibility.days <= 3
        ? { bg: "bg-amber-50", border: "border-amber-300", text: "text-amber-900", chipBg: "bg-amber-100", chipText: "text-amber-800" }
        : { bg: "bg-sky-50", border: "border-sky-300", text: "text-sky-900", chipBg: "bg-sky-100", chipText: "text-sky-700" };

  if (variant === "global") {
    function dismiss() {
      if (typeof window !== "undefined" && visibility) {
        const key = `renewal-dismissed:${visibility.periodEnd}`;
        window.localStorage.setItem(key, "1");
        setDismissedKey(key);
      }
    }
    return (
      <div className={`${tone.bg} border-b ${tone.border} ${tone.text} px-4 py-2.5`}>
        <div className="max-w-6xl mx-auto flex flex-wrap items-center gap-x-4 gap-y-1.5 text-sm">
          <span className={`${tone.chipBg} ${tone.chipText} text-[10px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded`}>
            {visibility.providerLabel} renewal
          </span>
          <span className="font-medium">
            Your {visibility.planName} plan{" "}
            {visibility.days <= 0 ? "has expired" : `expires ${urgency}`} —
            mobile-money subscriptions don&apos;t auto-renew.
          </span>
          <Link
            href={checkoutHref}
            className="ml-auto inline-flex items-center px-3 py-1 rounded-md bg-white shadow-sm font-semibold hover:bg-slate-50"
          >
            Renew now →
          </Link>
          <button
            onClick={dismiss}
            aria-label="Dismiss"
            className="opacity-60 hover:opacity-100 px-1"
          >
            ✕
          </button>
        </div>
      </div>
    );
  }

  // Inline (card) variant for /app/billing
  return (
    <div className={`rounded-2xl border ${tone.border} ${tone.bg} p-6`}>
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <div className={`text-[10px] uppercase tracking-wider font-bold ${tone.chipText}`}>
            Renewal required
          </div>
          <h2 className={`mt-1 text-xl font-bold ${tone.text}`}>
            Your {visibility.planName} plan {visibility.days <= 0 ? "has expired" : `expires ${urgency}`}
          </h2>
          <p className={`mt-1.5 text-sm ${tone.text}`}>
            {visibility.providerLabel} subscriptions don&apos;t auto-renew. Re-pay before{" "}
            <span className="font-mono font-semibold">
              {new Date(visibility.periodEnd).toLocaleDateString()}
            </span>{" "}
            to keep clinical screening uninterrupted.
          </p>
        </div>
        <Link
          href={checkoutHref}
          className="inline-flex items-center px-4 py-2 text-sm font-semibold rounded-lg bg-slate-900 hover:bg-slate-800 text-white"
        >
          Renew {visibility.planName}
        </Link>
      </div>
    </div>
  );
}
