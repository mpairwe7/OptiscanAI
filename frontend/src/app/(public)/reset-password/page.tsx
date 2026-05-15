import { Suspense } from "react";
import { ResetPasswordForm } from "@/components/auth/reset-password-form";

export const metadata = { title: "Reset password — OptiscanAI" };

export default function ResetPasswordPage() {
  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md">
        <div className="text-center">
          <h1 className="text-3xl font-bold tracking-tight text-slate-900">Reset password</h1>
        </div>
        <Suspense fallback={null}>
          <ResetPasswordForm />
        </Suspense>
      </div>
    </div>
  );
}
