"use client";
import { useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { apiForgotPassword, apiResetPassword } from "@/lib/auth-api";
import { ApiError } from "@/lib/api-fetch";

export function ResetPasswordForm() {
  const params = useSearchParams();
  const token = params.get("token");

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  async function handleForgot(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await apiForgotPassword(email);
      setDone(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleReset(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await apiResetPassword(token!, password);
      setDone(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed");
    } finally {
      setSubmitting(false);
    }
  }

  if (done) {
    return (
      <div className="mt-8 rounded-xl border border-slate-200 bg-white p-6 text-center">
        <div className="w-12 h-12 mx-auto rounded-full bg-teal-50 text-teal-700 flex items-center justify-center font-bold">✓</div>
        <h2 className="mt-4 font-semibold text-slate-900">
          {token ? "Password reset" : "Check your email"}
        </h2>
        <p className="mt-2 text-sm text-slate-600">
          {token
            ? "You can now sign in with your new password."
            : "If an account exists for that address, we sent a reset link. It expires in 1 hour."}
        </p>
        <Link
          href="/sign-in"
          className="mt-4 inline-flex items-center justify-center min-h-[44px] px-4 py-2 text-sm font-semibold rounded-lg bg-teal-700 hover:bg-teal-800 text-white transition-colors"
        >
          Sign in
        </Link>
      </div>
    );
  }

  if (token) {
    return (
      <form onSubmit={handleReset} className="mt-8 space-y-4 rounded-xl border border-slate-200 bg-white p-6">
        <div>
          <label htmlFor="reset-password" className="block text-sm font-medium text-slate-700">New password</label>
          <input
            id="reset-password"
            type="password"
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="new-password"
            className="mt-1 w-full px-3 py-2.5 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 text-sm"
          />
        </div>
        {error && <div className="rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">{error}</div>}
        <button
          type="submit"
          disabled={submitting}
          className="w-full inline-flex items-center justify-center min-h-[44px] px-4 py-2.5 text-sm font-semibold rounded-lg bg-teal-700 hover:bg-teal-800 text-white disabled:opacity-50 transition-colors shadow-sm"
        >
          {submitting ? "Resetting…" : "Reset password"}
        </button>
      </form>
    );
  }

  return (
    <form onSubmit={handleForgot} className="mt-8 space-y-4 rounded-xl border border-slate-200 bg-white p-6">
      <p className="text-sm text-slate-600">Enter the email tied to your account.</p>
      <div>
        <label htmlFor="forgot-email" className="block text-sm font-medium text-slate-700">Email</label>
        <input
          id="forgot-email"
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          autoComplete="email"
          className="mt-1 w-full px-3 py-2.5 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 text-sm"
        />
      </div>
      {error && <div className="rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">{error}</div>}
      <button
        type="submit"
        disabled={submitting}
        className="w-full inline-flex items-center justify-center min-h-[44px] px-4 py-2.5 text-sm font-semibold rounded-lg bg-teal-700 hover:bg-teal-800 text-white disabled:opacity-50 transition-colors shadow-sm"
      >
        {submitting ? "Sending…" : "Send reset link"}
      </button>
    </form>
  );
}
