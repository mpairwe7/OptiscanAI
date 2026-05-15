"use client";
import { useState } from "react";
import { PricingCard } from "@/components/billing/pricing-card";
import { BillingPeriodToggle } from "@/components/billing/billing-period-toggle";
import { PLANS, type BillingPeriod } from "@/lib/plans";

export function PricingCards() {
  const [period, setPeriod] = useState<BillingPeriod>("annual");

  return (
    <div className="mt-10">
      <div className="flex justify-center">
        <BillingPeriodToggle value={period} onChange={setPeriod} />
      </div>
      <div className="mt-8 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {PLANS.map((p) => (
          <PricingCard key={p.id} plan={p} period={period} />
        ))}
      </div>
    </div>
  );
}
