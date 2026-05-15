"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { apiJson } from "@/lib/api-fetch";

export function VerifyEmail() {
  const params = useSearchParams();
  const token = params.get("token");
  const [state, setState] = useState<"verifying" | "ok" | "error">("verifying");
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!token) {
      setState("error");
      setMessage("Missing token.");
      return;
    }
    (async () => {
      try {
        await apiJson<{ status: string }>(
          `/api/v1/auth/verify-email?token=${encodeURIComponent(token)}`,
          { silent: true },
        );
        setState("ok");
      } catch (err) {
        setState("error");
        setMessage(err instanceof Error ? err.message : "Verification failed.");
      }
    })();
  }, [token]);

  return (
    <div className="w-full max-w-md rounded-xl border border-slate-200 bg-white p-6 text-center">
      {state === "verifying" && (
        <>
          <div className="w-10 h-10 mx-auto skeleton rounded-full" />
          <h1 className="mt-4 font-semibold text-slate-900">Verifying…</h1>
        </>
      )}
      {state === "ok" && (
        <>
          <div className="w-10 h-10 mx-auto rounded-full bg-teal-50 text-teal-600 flex items-center justify-center">✓</div>
          <h1 className="mt-4 font-semibold text-slate-900">Email verified</h1>
          <Link
            href="/app/dashboard"
            className="mt-4 inline-flex items-center justify-center px-4 py-2 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white"
          >
            Continue to dashboard
          </Link>
        </>
      )}
      {state === "error" && (
        <>
          <div className="w-10 h-10 mx-auto rounded-full bg-red-50 text-red-600 flex items-center justify-center">!</div>
          <h1 className="mt-4 font-semibold text-slate-900">Couldn&apos;t verify email</h1>
          <p className="mt-1.5 text-sm text-slate-600">{message}</p>
          <Link
            href="/sign-in"
            className="mt-4 inline-flex items-center justify-center px-4 py-2 text-sm font-semibold rounded-lg border border-slate-300 hover:bg-slate-50 text-slate-700"
          >
            Back to sign-in
          </Link>
        </>
      )}
    </div>
  );
}
