"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import {
  apiFlutterwaveCheckout,
  apiMomoCheckout,
  apiPollIntent,
  apiStripeCheckout,
  type PaymentIntentDTO,
} from "@/lib/auth-api";
import { ApiError } from "@/lib/api-fetch";
import {
  annualSavingsLabel,
  formatPrice,
  planById,
  type BillingPeriod,
  type PlanId,
} from "@/lib/plans";

type Method = "card" | "mtn" | "airtel" | "flutterwave";

const METHODS: { id: Method; label: string; subtitle: string }[] = [
  { id: "card", label: "Card", subtitle: "Visa, Mastercard, Apple Pay (Stripe)" },
  { id: "mtn", label: "MTN MoMo", subtitle: "Mobile money push (Uganda)" },
  { id: "airtel", label: "Airtel Money", subtitle: "Mobile money push (Uganda)" },
  { id: "flutterwave", label: "Flutterwave", subtitle: "Pan-African cards + MoMo" },
];

export function CheckoutForm({ planCode }: { planCode: string }) {
  const params = useSearchParams();
  const router = useRouter();
  const qc = useQueryClient();
  const cycle = (params.get("cycle") as BillingPeriod) || "monthly";
  const plan = planById(planCode as PlanId);

  const [method, setMethod] = useState<Method>("card");
  const [phone, setPhone] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingIntent, setPendingIntent] = useState<PaymentIntentDTO | null>(null);

  // Poll MoMo intent status every 3s until terminal
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if (!pendingIntent) return;
    if (pendingIntent.status === "succeeded") {
      qc.invalidateQueries({ queryKey: ["auth", "me"] });
      qc.invalidateQueries({ queryKey: ["billing"] });
      router.push("/app/checkout/success");
      return;
    }
    if (["failed", "canceled"].includes(pendingIntent.status)) {
      setError("Payment failed or was canceled. Please try again.");
      setPendingIntent(null);
      return;
    }
    pollRef.current = setInterval(async () => {
      try {
        const res = await apiPollIntent(pendingIntent.id);
        setPendingIntent(res);
      } catch {
        // ignore transient errors; user can refresh
      }
    }, 3000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [pendingIntent, qc, router]);

  if (!plan) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-6">
        <h1 className="font-bold text-slate-900">Plan not found</h1>
        <Link href="/pricing" className="mt-2 inline-flex text-teal-600 hover:text-teal-700">
          Back to pricing
        </Link>
      </div>
    );
  }

  if (plan.id === "free") {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-6">
        <h1 className="font-bold text-slate-900">Free plan</h1>
        <p className="mt-1.5 text-sm text-slate-600">You&apos;re already on the free plan.</p>
        <Link
          href="/app/screening"
          className="mt-4 inline-flex items-center px-4 py-2 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white"
        >
          Start screening
        </Link>
      </div>
    );
  }

  if (plan.id === "health_system") {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-6">
        <h1 className="font-bold text-slate-900">Talk to sales</h1>
        <p className="mt-1.5 text-sm text-slate-600">Health System pricing is custom.</p>
        <Link
          href="/contact-sales"
          className="mt-4 inline-flex items-center px-4 py-2 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white"
        >
          Contact sales
        </Link>
      </div>
    );
  }

  async function handleSubscribe() {
    if (!plan) return;
    setSubmitting(true);
    setError(null);
    try {
      if (method === "card") {
        const res = await apiStripeCheckout(plan.id as "clinician" | "practice", cycle);
        window.location.href = res.url;
        return;
      }
      if (method === "mtn" || method === "airtel") {
        if (!/^\+?\d{10,}$/.test(phone.replace(/\s+/g, ""))) {
          setError("Enter a valid phone number (Uganda: +256… or 07…).");
          return;
        }
        const res = await apiMomoCheckout(
          plan.id as "clinician" | "practice",
          cycle,
          phone,
          method,
        );
        setPendingIntent({
          id: res.intent_id,
          status: res.status as PaymentIntentDTO["status"],
          provider: res.provider,
          plan_code: plan.id,
          billing_cycle: cycle,
          amount_cents: 0,
          currency: "USD",
          confirmed_at: null,
        });
        return;
      }
      if (method === "flutterwave") {
        const res = await apiFlutterwaveCheckout(
          plan.id as "clinician" | "practice",
          cycle,
          phone,
        );
        window.location.href = res.payment_url;
        return;
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Checkout failed");
    } finally {
      setSubmitting(false);
    }
  }

  const price = formatPrice(plan, cycle);
  const savings = cycle === "annual" ? annualSavingsLabel(plan) : null;

  if (pendingIntent && ["requires_action", "processing"].includes(pendingIntent.status)) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-8 text-center">
        <div className="w-12 h-12 mx-auto skeleton rounded-full" />
        <h1 className="mt-4 text-xl font-bold text-slate-900">Awaiting payment confirmation</h1>
        <p className="mt-1.5 text-sm text-slate-600">
          A push prompt has been sent to <span className="font-mono">{phone}</span>. Enter your PIN
          on your phone to confirm. This page will update automatically.
        </p>
        <p className="mt-3 text-xs text-slate-500">
          Provider: <span className="capitalize font-semibold text-slate-700">{pendingIntent.provider}</span> ·
          Intent <span className="font-mono">{pendingIntent.id.slice(0, 8)}…</span>
        </p>
        <button
          onClick={() => {
            if (pollRef.current) clearInterval(pollRef.current);
            setPendingIntent(null);
          }}
          className="mt-6 inline-flex items-center px-4 py-2 text-sm font-medium rounded-lg border border-slate-300 hover:bg-slate-50 text-slate-700"
        >
          Cancel and choose another method
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900">Subscribe to {plan.name}</h1>
        <p className="mt-1 text-sm text-slate-600">{plan.tagline}</p>
      </header>

      <section className="rounded-2xl border border-slate-200 bg-white p-6">
        <h2 className="font-semibold text-slate-900">Order summary</h2>
        <div className="mt-4 grid sm:grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-slate-500">Plan</div>
            <div className="mt-0.5 font-semibold text-slate-900">{plan.name}</div>
          </div>
          <div>
            <div className="text-slate-500">Billing cycle</div>
            <div className="mt-0.5 font-semibold text-slate-900 capitalize">{cycle}</div>
          </div>
          <div>
            <div className="text-slate-500">Price</div>
            <div className="mt-0.5 font-semibold text-slate-900">{price}</div>
            {savings && <div className="mt-0.5 text-xs text-teal-600">{savings}</div>}
          </div>
          <div>
            <div className="text-slate-500">Includes</div>
            <div className="mt-0.5 text-slate-900">
              {typeof plan.scanQuota === "number"
                ? `${plan.scanQuota.toLocaleString()} scans/mo`
                : "Unlimited scans"}
              {typeof plan.seats === "number" && plan.seats > 1 && ` · ${plan.seats} seats`}
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-6">
        <h2 className="font-semibold text-slate-900">Payment method</h2>
        <div className="mt-4 grid sm:grid-cols-2 gap-3">
          {METHODS.map((m) => (
            <button
              key={m.id}
              onClick={() => setMethod(m.id)}
              className={`text-left rounded-xl border p-4 transition-colors ${
                method === m.id
                  ? "border-teal-500 ring-2 ring-teal-500/20 bg-teal-50/30"
                  : "border-slate-200 hover:bg-slate-50"
              }`}
            >
              <div className="font-semibold text-slate-900">{m.label}</div>
              <div className="mt-0.5 text-xs text-slate-500">{m.subtitle}</div>
            </button>
          ))}
        </div>

        {(method === "mtn" || method === "airtel" || method === "flutterwave") && (
          <div className="mt-5">
            <label className="block text-sm font-medium text-slate-700">
              {method === "flutterwave" ? "Phone (optional, for MoMo on Flutterwave)" : "Phone number"}
            </label>
            <input
              type="tel"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              placeholder="+256 7… (Uganda)"
              autoComplete="tel"
              className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            />
            {(method === "mtn" || method === "airtel") && (
              <p className="mt-1 text-xs text-slate-500">
                We&apos;ll send a USSD prompt to this number. Enter your MoMo PIN to confirm.
              </p>
            )}
          </div>
        )}

        {error && (
          <div className="mt-4 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">
            {error}
          </div>
        )}

        <button
          onClick={handleSubscribe}
          disabled={submitting}
          className="mt-6 w-full inline-flex items-center justify-center px-4 py-2.5 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white disabled:opacity-50"
        >
          {submitting
            ? "Processing…"
            : method === "card"
              ? `Subscribe — ${price}`
              : method === "flutterwave"
                ? `Continue with Flutterwave — ${price}`
                : `Send payment prompt — ${price}`}
        </button>

        <p className="mt-3 text-xs text-slate-500">
          {method === "card" && "Secured by Stripe (PCI DSS Level 1). Cancel any time from Billing."}
          {method === "flutterwave" && "Redirects to Flutterwave's hosted payment page."}
          {(method === "mtn" || method === "airtel") &&
            "Charged in UGX at the current FX rate. MoMo subscriptions don't auto-renew — we'll email a reminder before your period ends."}
        </p>
      </section>

      <div className="flex flex-wrap items-center gap-3 justify-between">
        <Link href="/pricing" className="text-sm text-slate-600 hover:text-slate-900">
          ← Back to pricing
        </Link>
      </div>
    </div>
  );
}
