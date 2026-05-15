"use client";
import { useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { apiMe, apiSignOut } from "@/lib/auth-api";
import { useAuthStore } from "@/stores/auth-store";

export default function AccountPage() {
  const router = useRouter();
  const setUser = useAuthStore((s) => s.setUser);
  const me = useQuery({ queryKey: ["auth", "me"], queryFn: apiMe });

  async function handleSignOut() {
    await apiSignOut();
    setUser(null);
    router.push("/");
  }

  if (me.isLoading || !me.data) {
    return <div className="skeleton h-32 rounded-2xl" />;
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900">Account</h1>
        <p className="mt-1 text-sm text-slate-600">Manage your profile, email, and session.</p>
      </header>

      <section className="rounded-2xl border border-slate-200 bg-white p-6">
        <h2 className="font-semibold text-slate-900">Profile</h2>
        <dl className="mt-4 grid sm:grid-cols-2 gap-4 text-sm">
          <div>
            <dt className="text-slate-500">Full name</dt>
            <dd className="mt-0.5 text-slate-900">{me.data.full_name || "—"}</dd>
          </div>
          <div>
            <dt className="text-slate-500">Email</dt>
            <dd className="mt-0.5 text-slate-900 flex items-center gap-2">
              {me.data.email}
              {me.data.email_verified ? (
                <span className="text-[10px] uppercase tracking-wider font-bold text-teal-600 bg-teal-50 px-1.5 py-0.5 rounded">
                  verified
                </span>
              ) : (
                <span className="text-[10px] uppercase tracking-wider font-bold text-amber-600 bg-amber-50 px-1.5 py-0.5 rounded">
                  unverified
                </span>
              )}
            </dd>
          </div>
          <div>
            <dt className="text-slate-500">Organization</dt>
            <dd className="mt-0.5 text-slate-900">{me.data.organization.name}</dd>
          </div>
          <div>
            <dt className="text-slate-500">Role</dt>
            <dd className="mt-0.5 text-slate-900 capitalize">{me.data.role}</dd>
          </div>
        </dl>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-6">
        <h2 className="font-semibold text-slate-900">Session</h2>
        <p className="mt-1 text-sm text-slate-600">Sign out of all devices.</p>
        <button
          onClick={handleSignOut}
          className="mt-4 inline-flex items-center px-4 py-2 text-sm font-semibold rounded-lg border border-slate-300 hover:bg-slate-50 text-slate-700"
        >
          Sign out
        </button>
      </section>
    </div>
  );
}
