import { Suspense } from "react";
import { VerifyEmail } from "@/components/auth/verify-email";

export const metadata = { title: "Verify email — OptiscanAI" };

export default function VerifyEmailPage() {
  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4 py-12">
      <Suspense fallback={null}>
        <VerifyEmail />
      </Suspense>
    </div>
  );
}
