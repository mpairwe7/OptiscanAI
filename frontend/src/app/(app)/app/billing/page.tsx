"use client";
import { useState } from "react";
import Link from "next/link";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  apiCancelSubscription,
  apiGetSubscription,
  apiListInvoices,
  apiResumeSubscription,
  apiStripePortal,
} from "@/lib/auth-api";
import { ApiError } from "@/lib/api-fetch";
import { RenewalBanner } from "@/components/billing/renewal-banner";
import { DowngradeDialog } from "@/components/billing/downgrade-dialog";

export default function BillingPage() {
  const qc = useQueryClient();
  const sub = useQuery({ queryKey: ["billing", "subscription"], queryFn: apiGetSubscription });
  const invoices = useQuery({ queryKey: ["billing", "invoices"], queryFn: apiListInvoices });
  const [portalLoading, setPortalLoading] = useState(false);
  const [portalError, setPortalError] = useState<string | null>(null);
  const [downgrading, setDowngrading] = useState(false);

  async function handlePortal() {
    setPortalLoading(true);
    setPortalError(null);
    try {
      const { url } = await apiStripePortal();
      window.location.href = url;
    } catch (err) {
      setPortalError(err instanceof ApiError ? err.message : "Portal failed");
    } finally {
      setPortalLoading(false);
    }
  }

  async function handleCancel() {
    if (!confirm("Cancel at end of current period? You'll keep access until then.")) return;
    await apiCancelSubscription();
    qc.invalidateQueries({ queryKey: ["billing"] });
    qc.invalidateQueries({ queryKey: ["auth", "me"] });
  }

  async function handleResume() {
    await apiResumeSubscription();
    qc.invalidateQueries({ queryKey: ["billing"] });
    qc.invalidateQueries({ queryKey: ["auth", "me"] });
  }

  if (sub.isLoading) return <div className="skeleton h-40 rounded-2xl" />;
  if (!sub.data) return <p>No subscription.</p>;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900">Billing</h1>
        <p className="mt-1 text-sm text-slate-600">Manage your plan and payment history.</p>
      </header>

      <RenewalBanner variant="inline" />

      <section className="rounded-2xl border border-slate-200 bg-white p-6">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <div className="text-[10px] uppercase tracking-wider font-bold text-slate-500">Current plan</div>
            <h2 className="mt-1 text-2xl font-bold text-slate-900">{sub.data.plan_display_name}</h2>
            <div className="mt-1 text-sm text-slate-600">
              {sub.data.billing_cycle === "annual" ? "Billed annually" : "Billed monthly"} ·{" "}
              <span className="capitalize">{sub.data.status}</span>
              {sub.data.cancel_at_period_end && " · Cancels at period end"}
            </div>
            <div className="mt-1 text-xs text-slate-500">
              Renews on {new Date(sub.data.current_period_end).toLocaleDateString()}
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Link
              href="/pricing"
              className="inline-flex items-center px-3 py-1.5 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white"
            >
              Change plan
            </Link>
            {sub.data.provider === "stripe" && (
              <button
                onClick={handlePortal}
                disabled={portalLoading}
                className="inline-flex items-center px-3 py-1.5 text-sm font-semibold rounded-lg border border-slate-300 hover:bg-slate-50 text-slate-700 disabled:opacity-50"
              >
                {portalLoading ? "Opening…" : "Manage payment method"}
              </button>
            )}
            {sub.data.cancel_at_period_end ? (
              <button
                onClick={handleResume}
                className="inline-flex items-center px-3 py-1.5 text-sm font-semibold rounded-lg border border-slate-300 hover:bg-slate-50 text-slate-700"
              >
                Resume
              </button>
            ) : (
              sub.data.plan_code !== "free" && (
                <>
                  <button
                    onClick={handleCancel}
                    className="inline-flex items-center px-3 py-1.5 text-sm font-semibold rounded-lg border border-slate-300 hover:bg-slate-50 text-slate-700"
                  >
                    Cancel at period end
                  </button>
                  <button
                    onClick={() => setDowngrading(true)}
                    className="inline-flex items-center px-3 py-1.5 text-sm font-medium rounded-lg text-red-600 hover:text-red-700 hover:bg-red-50"
                  >
                    Downgrade to Free
                  </button>
                </>
              )
            )}
          </div>
        </div>
        {portalError && (
          <div className="mt-3 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">
            {portalError}
          </div>
        )}
      </section>

      {downgrading && (
        <DowngradeDialog
          currentPlanName={sub.data.plan_display_name}
          currentProvider={sub.data.provider}
          onClose={() => setDowngrading(false)}
        />
      )}

      <section className="rounded-2xl border border-slate-200 bg-white p-6">
        <h2 className="font-semibold text-slate-900">Invoices</h2>
        {!invoices.data || invoices.data.length === 0 ? (
          <p className="mt-3 text-sm text-slate-500">No invoices yet.</p>
        ) : (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-slate-500">
                  <th className="py-2 pr-4 font-medium">Date</th>
                  <th className="py-2 pr-4 font-medium">Amount</th>
                  <th className="py-2 pr-4 font-medium">Status</th>
                  <th className="py-2 pr-4 font-medium">Method</th>
                  <th className="py-2 pr-4 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {invoices.data.map((inv) => (
                  <tr key={inv.id} className="border-b border-slate-100">
                    <td className="py-2 pr-4 text-slate-900">
                      {new Date(inv.issued_at).toLocaleDateString()}
                    </td>
                    <td className="py-2 pr-4 text-slate-900 font-mono">
                      ${(inv.amount_cents / 100).toFixed(2)}
                    </td>
                    <td className="py-2 pr-4 capitalize text-slate-700">{inv.status}</td>
                    <td className="py-2 pr-4 capitalize text-slate-700">{inv.provider}</td>
                    <td className="py-2 pr-4">
                      {inv.pdf_url && (
                        <a href={inv.pdf_url} className="text-teal-600 hover:text-teal-700" target="_blank" rel="noreferrer">
                          PDF
                        </a>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
