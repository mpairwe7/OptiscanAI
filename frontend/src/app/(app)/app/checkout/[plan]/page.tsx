import { Suspense } from "react";
import { CheckoutForm } from "@/components/billing/checkout-form";

interface Props {
  params: Promise<{ plan: string }>;
}

export const metadata = { title: "Checkout — OptiscanAI" };

export default async function CheckoutPage({ params }: Props) {
  const { plan } = await params;
  return (
    <div className="max-w-2xl mx-auto">
      <Suspense fallback={null}>
        <CheckoutForm planCode={plan} />
      </Suspense>
    </div>
  );
}
