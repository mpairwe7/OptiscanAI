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
  const [showPassword, setShowPassword] = useState(false);
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
        <div className="w-12 h-12 mx-auto rounded-full bg-teal-50 text-teal-700 flex items-center justify-center font-bold">
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
        <label htmlFor="signin-email" className="block text-sm font-medium text-slate-700">
          Email
        </label>
        <input
          id="signin-email"
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          autoComplete="email"
          className="mt-1 w-full px-3 py-2.5 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 text-base sm:text-sm"
        />
      </div>

      {mode === "password" && (
        <div>
          <div className="flex items-baseline justify-between">
            <label htmlFor="signin-password" className="block text-sm font-medium text-slate-700">
              Password
            </label>
            <Link href="/reset-password" className="text-xs font-medium text-teal-700 hover:text-teal-800 py-1">
              Forgot?
            </Link>
          </div>
          <div className="relative mt-1">
            <input
              id="signin-password"
              type={showPassword ? "text" : "password"}
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              className="w-full pl-3 pr-10 py-2.5 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 text-base sm:text-sm"
            />
            <button
              type="button"
              onClick={() => setShowPassword((prev) => !prev)}
              aria-label={showPassword ? "Hide password" : "Show password"}
              className="absolute inset-y-0 right-0 flex items-center pr-3 text-slate-400 hover:text-slate-600 focus:outline-none focus:text-teal-600"
            >
              {showPassword ? (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                  className="w-5 h-5"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3.98 8.223A10.477 10.477 0 0 0 1.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.451 10.451 0 0 1 12 4.5c4.756 0 8.773 3.162 10.065 7.498a10.522 10.522 0 0 1-4.293 5.774M6.228 6.228 3 3m3.228 3.228 3.65 3.65m7.894 7.894L21 21m-3.228-3.228-3.65-3.65m0 0a3 3 0 1 0-4.243-4.243m4.242 4.242L9.88 9.88"
                  />
                </svg>
              ) : (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                  className="w-5 h-5"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z"
                  />
                </svg>
              )}
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">{error}</div>
      )}

      <button
        type="submit"
        disabled={submitting}
        className="w-full inline-flex items-center justify-center min-h-[44px] px-4 py-2.5 text-sm font-semibold rounded-lg bg-teal-700 hover:bg-teal-800 text-white disabled:opacity-50 transition-colors shadow-sm"
      >
        {submitting ? "Signing in…" : mode === "password" ? "Sign in" : "Send magic link"}
      </button>

      <button
        type="button"
        onClick={() => {
          setMode(mode === "password" ? "magic" : "password");
          setError(null);
        }}
        className="w-full text-center py-2 text-xs font-medium text-slate-600 hover:text-slate-900"
      >
        {mode === "password" ? "Sign in with a magic link instead" : "Use password instead"}
      </button>

      <div className="text-center text-sm text-slate-600 pt-2 border-t border-slate-100">
        New here?{" "}
        <Link href="/sign-up" className="font-semibold text-teal-700 hover:text-teal-800 py-1">
          Create an account
        </Link>
      </div>
    </form>
  );
}
