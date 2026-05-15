import { Suspense } from "react";
import { SignInForm } from "@/components/auth/sign-in-form";

export const metadata = { title: "Sign in — OptiscanAI" };

export default function SignInPage() {
  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md">
        <div className="text-center">
          <h1 className="text-3xl font-bold tracking-tight text-slate-900">Welcome back</h1>
          <p className="mt-2 text-sm text-slate-600">Sign in to continue to OptiscanAI.</p>
        </div>
        <Suspense fallback={null}>
          <SignInForm />
        </Suspense>
      </div>
    </div>
  );
}
