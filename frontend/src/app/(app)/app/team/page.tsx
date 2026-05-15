"use client";
import { useState } from "react";
import Link from "next/link";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  apiInviteMember,
  apiListInvites,
  apiListMembers,
  apiMe,
  apiRemoveMember,
  apiRevokeInvite,
  apiUpdateMemberRole,
} from "@/lib/auth-api";
import { ApiError } from "@/lib/api-fetch";
import { SeatManager } from "@/components/team/seat-manager";

export default function TeamPage() {
  const qc = useQueryClient();
  const me = useQuery({ queryKey: ["auth", "me"], queryFn: apiMe });
  const orgId = me.data?.organization.id;

  const members = useQuery({
    queryKey: ["orgs", orgId, "members"],
    queryFn: () => apiListMembers(orgId!),
    enabled: !!orgId,
  });
  const invites = useQuery({
    queryKey: ["orgs", orgId, "invites"],
    queryFn: () => apiListInvites(orgId!),
    enabled: !!orgId,
  });

  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<"admin" | "clinician" | "viewer">("clinician");
  const [inviteError, setInviteError] = useState<string | null>(null);
  const [inviting, setInviting] = useState(false);

  async function handleInvite(e: React.FormEvent) {
    e.preventDefault();
    if (!orgId) return;
    setInviting(true);
    setInviteError(null);
    try {
      await apiInviteMember(orgId, inviteEmail, inviteRole);
      setInviteEmail("");
      qc.invalidateQueries({ queryKey: ["orgs", orgId, "invites"] });
    } catch (err) {
      setInviteError(err instanceof ApiError ? err.message : "Invite failed");
    } finally {
      setInviting(false);
    }
  }

  const plan = me.data?.subscription?.plan;
  const tier = plan?.code;
  const allowed = tier === "practice" || tier === "health_system";

  if (me.isLoading) return <div className="skeleton h-40 rounded-2xl" />;

  if (!allowed) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-6 max-w-md">
        <h1 className="text-xl font-bold text-slate-900">Team management</h1>
        <p className="mt-1.5 text-sm text-slate-600">
          Multi-seat team management is a Practice feature. Upgrade to invite clinicians and share a review queue.
        </p>
        <Link
          href="/app/checkout/practice?cycle=monthly"
          className="mt-4 inline-flex items-center px-4 py-2 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white"
        >
          Upgrade to Practice
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900">Team</h1>
        <p className="mt-1 text-sm text-slate-600">Invite clinicians and manage your seat allowance.</p>
      </header>

      <SeatManager />

      <form
        onSubmit={handleInvite}
        className="rounded-2xl border border-slate-200 bg-white p-6 grid sm:grid-cols-[1fr_auto_auto] gap-3 items-end"
      >
        <div>
          <label className="block text-sm font-medium text-slate-700">Invite email</label>
          <input
            type="email"
            required
            value={inviteEmail}
            onChange={(e) => setInviteEmail(e.target.value)}
            className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            placeholder="clinician@hospital.org"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700">Role</label>
          <select
            value={inviteRole}
            onChange={(e) => setInviteRole(e.target.value as "admin" | "clinician" | "viewer")}
            className="mt-1 px-3 py-2 rounded-lg border border-slate-300"
          >
            <option value="admin">Admin</option>
            <option value="clinician">Clinician</option>
            <option value="viewer">Viewer</option>
          </select>
        </div>
        <button
          type="submit"
          disabled={inviting}
          className="inline-flex items-center justify-center px-4 py-2 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white disabled:opacity-50"
        >
          {inviting ? "Sending…" : "Send invite"}
        </button>
        {inviteError && (
          <div className="sm:col-span-3 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">
            {inviteError}
          </div>
        )}
      </form>

      <section className="rounded-2xl border border-slate-200 bg-white p-6">
        <h2 className="font-semibold text-slate-900">Members</h2>
        <table className="mt-4 w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-slate-500">
              <th className="py-2 font-medium">Name</th>
              <th className="py-2 font-medium">Email</th>
              <th className="py-2 font-medium">Role</th>
              <th className="py-2 font-medium">Joined</th>
              <th className="py-2 font-medium" />
            </tr>
          </thead>
          <tbody>
            {(members.data ?? []).map((m) => (
              <tr key={m.user_id} className="border-b border-slate-100">
                <td className="py-2 text-slate-900">{m.full_name || "—"}</td>
                <td className="py-2 text-slate-700">{m.email}</td>
                <td className="py-2">
                  {m.role === "owner" ? (
                    <span className="capitalize text-slate-700">{m.role}</span>
                  ) : (
                    <select
                      defaultValue={m.role}
                      onChange={async (e) => {
                        await apiUpdateMemberRole(orgId!, m.user_id, e.target.value as "admin" | "clinician" | "viewer");
                        qc.invalidateQueries({ queryKey: ["orgs", orgId, "members"] });
                      }}
                      className="px-2 py-1 rounded-lg border border-slate-300 text-sm"
                    >
                      <option value="admin">Admin</option>
                      <option value="clinician">Clinician</option>
                      <option value="viewer">Viewer</option>
                    </select>
                  )}
                </td>
                <td className="py-2 text-slate-500">
                  {m.joined_at ? new Date(m.joined_at).toLocaleDateString() : "—"}
                </td>
                <td className="py-2 text-right">
                  {m.role !== "owner" && (
                    <button
                      onClick={async () => {
                        if (!confirm(`Remove ${m.email}?`)) return;
                        await apiRemoveMember(orgId!, m.user_id);
                        qc.invalidateQueries({ queryKey: ["orgs", orgId, "members"] });
                      }}
                      className="text-xs text-red-600 hover:text-red-700"
                    >
                      Remove
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {invites.data && invites.data.length > 0 && (
        <section className="rounded-2xl border border-slate-200 bg-white p-6">
          <h2 className="font-semibold text-slate-900">Pending invites</h2>
          <table className="mt-4 w-full text-sm">
            <tbody>
              {invites.data.map((i) => (
                <tr key={i.id} className="border-b border-slate-100">
                  <td className="py-2 text-slate-900">{i.email}</td>
                  <td className="py-2 capitalize text-slate-700">{i.role}</td>
                  <td className="py-2 text-slate-500">
                    Expires {new Date(i.expires_at).toLocaleDateString()}
                  </td>
                  <td className="py-2 text-right">
                    <button
                      onClick={async () => {
                        await apiRevokeInvite(orgId!, i.id);
                        qc.invalidateQueries({ queryKey: ["orgs", orgId, "invites"] });
                      }}
                      className="text-xs text-red-600 hover:text-red-700"
                    >
                      Revoke
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
