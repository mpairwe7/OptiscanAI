import { Suspense } from "react";
import { SignUpForm } from "@/components/auth/sign-up-form";

export const metadata = { title: "Get started — OptiscanAI" };

export default function SignUpPage() {
  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md">
        <div className="text-center">
          <h1 className="text-3xl font-bold tracking-tight text-slate-900">Create your account</h1>
          <p className="mt-2 text-sm text-slate-600">
            10 free scans / month. No credit card required.
          </p>
        </div>
        <Suspense fallback={null}>
          <SignUpForm />
        </Suspense>
      </div>
    </div>
  );
}
