"use client";
import { useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import { apiSignUp } from "@/lib/auth-api";
import { useAuthStore } from "@/stores/auth-store";
import { ApiError } from "@/lib/api-fetch";

export function SignUpForm() {
  const router = useRouter();
  const params = useSearchParams();
  const intendedPlan = params.get("plan");
  const next = params.get("next") || (intendedPlan && intendedPlan !== "free" ? `/app/checkout/${intendedPlan}` : "/app/dashboard");
  const qc = useQueryClient();
  const setUser = useAuthStore((s) => s.setUser);

  const [email, setEmail] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const res = await apiSignUp({ email, password, full_name: fullName || undefined });
      setUser(res.user);
      qc.invalidateQueries({ queryKey: ["auth", "me"] });
      router.push(next);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Sign-up failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="mt-8 space-y-4 rounded-xl border border-slate-200 bg-white p-6">
      <div>
        <label className="block text-sm font-medium text-slate-700">Full name</label>
        <input
          type="text"
          value={fullName}
          onChange={(e) => setFullName(e.target.value)}
          autoComplete="name"
          placeholder="Dr Jane Doe"
          className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-slate-700">Work email</label>
        <input
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          autoComplete="email"
          className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-slate-700">Password</label>
        <input
          type="password"
          required
          minLength={8}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="new-password"
          className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
        />
        <p className="mt-1 text-xs text-slate-500">8+ characters.</p>
      </div>

      {error && (
        <div className="rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">{error}</div>
      )}

      <button
        type="submit"
        disabled={submitting}
        className="w-full inline-flex items-center justify-center px-4 py-2.5 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white disabled:opacity-50"
      >
        {submitting ? "Creating account…" : "Create account"}
      </button>

      <p className="text-center text-xs text-slate-500">
        By signing up you agree to our{" "}
        <Link href="/legal/terms" className="underline">terms</Link> and{" "}
        <Link href="/legal/privacy" className="underline">privacy policy</Link>.
      </p>

      <div className="text-center text-sm text-slate-600">
        Already have an account?{" "}
        <Link href="/sign-in" className="font-semibold text-teal-600 hover:text-teal-700">
          Sign in
        </Link>
      </div>
    </form>
  );
}
