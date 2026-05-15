"use client";
import { useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import { apiRequestMagicLink, apiSignIn } from "@/lib/auth-api";
import { useAuthStore } from "@/stores/auth-store";
import { ApiError } from "@/lib/api-fetch";

export function SignInForm() {
  const router = useRouter();
  const params = useSearchParams();
  const next = params.get("next") || "/app/dashboard";
  const qc = useQueryClient();
  const setUser = useAuthStore((s) => s.setUser);

  const [mode, setMode] = useState<"password" | "magic">("password");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [magicSent, setMagicSent] = useState(false);

  async function handlePasswordSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const res = await apiSignIn(email, password);
      setUser(res.user);
      qc.invalidateQueries({ queryKey: ["auth", "me"] });
      router.push(next);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Sign-in failed");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleMagicSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await apiRequestMagicLink(email);
      setMagicSent(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to send magic link");
    } finally {
      setSubmitting(false);
    }
  }

  if (magicSent) {
    return (
      <div className="mt-8 rounded-xl border border-slate-200 bg-white p-6 text-center">
        <div className="w-12 h-12 mx-auto rounded-full bg-teal-50 text-teal-600 flex items-center justify-center">
          ✉️
        </div>
        <h2 className="mt-4 font-semibold text-slate-900">Check your email</h2>
        <p className="mt-2 text-sm text-slate-600">
          If an account exists for <span className="font-medium">{email}</span>, we sent a sign-in link.
          It expires in 15 minutes.
        </p>
      </div>
    );
  }

  return (
    <form
      onSubmit={mode === "password" ? handlePasswordSubmit : handleMagicSubmit}
      className="mt-8 space-y-4 rounded-xl border border-slate-200 bg-white p-6"
    >
      <div>
        <label className="block text-sm font-medium text-slate-700">Email</label>
        <input
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          autoComplete="email"
          className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
        />
      </div>

      {mode === "password" && (
        <div>
          <div className="flex items-baseline justify-between">
            <label className="block text-sm font-medium text-slate-700">Password</label>
            <Link href="/reset-password" className="text-xs text-teal-600 hover:text-teal-700">
              Forgot?
            </Link>
          </div>
          <input
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
          />
        </div>
      )}

      {error && (
        <div className="rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">{error}</div>
      )}

      <button
        type="submit"
        disabled={submitting}
        className="w-full inline-flex items-center justify-center px-4 py-2.5 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white disabled:opacity-50"
      >
        {submitting ? "Signing in…" : mode === "password" ? "Sign in" : "Send magic link"}
      </button>

      <button
        type="button"
        onClick={() => {
          setMode(mode === "password" ? "magic" : "password");
          setError(null);
        }}
        className="w-full text-center text-xs text-slate-600 hover:text-slate-900"
      >
        {mode === "password" ? "Sign in with a magic link instead" : "Use password instead"}
      </button>

      <div className="text-center text-sm text-slate-600">
        New here?{" "}
        <Link href="/sign-up" className="font-semibold text-teal-600 hover:text-teal-700">
          Create an account
        </Link>
      </div>
    </form>
  );
}
