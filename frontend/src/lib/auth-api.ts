"use client";
/** Auth + billing + orgs API client (talks to FastAPI via same-origin rewrite). */
import { apiFetch, apiJson } from "@/lib/api-fetch";
import type { MeUser } from "@/stores/auth-store";

const API = ""; // same-origin via next.config.ts rewrite

// ── Auth ──

export async function apiSignIn(email: string, password: string) {
  return apiJson<{ user: MeUser; access_token: string; expires_in: number }>(
    `${API}/api/v1/auth/login`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
      silent: true,
    },
  );
}

export async function apiSignUp(input: { email: string; password: string; full_name?: string }) {
  return apiJson<{ user: MeUser; access_token: string; expires_in: number }>(
    `${API}/api/v1/auth/register`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
      silent: true,
    },
  );
}

export async function apiSignOut() {
  await apiFetch(`${API}/api/v1/auth/logout`, { method: "POST", silent: true });
}

export async function apiRequestMagicLink(email: string) {
  return apiJson<{ status: string }>(`${API}/api/v1/auth/magic-link/request`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
    silent: true,
  });
}

export async function apiForgotPassword(email: string) {
  return apiJson<{ status: string }>(`${API}/api/v1/auth/password/forgot`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
    silent: true,
  });
}

export async function apiResetPassword(token: string, new_password: string) {
  return apiJson<{ status: string }>(`${API}/api/v1/auth/password/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, new_password }),
    silent: true,
  });
}

export async function apiMe() {
  return apiJson<MeUser>(`${API}/api/v1/auth/me`, { silent: true });
}

// ── Billing ──

export interface PlanDTO {
  code: string;
  display_name: string;
  description: string | null;
  tagline: string | null;
  monthly_price_cents: number | null;
  annual_price_cents: number | null;
  currency: string;
  scan_limit_monthly: number | null;
  seat_limit: number | null;
  is_contact_sales: boolean;
  is_featured: boolean;
  features: Record<string, unknown>;
}

export async function apiListPlans() {
  return apiJson<PlanDTO[]>(`${API}/api/v1/billing/plans`);
}

export interface SubscriptionDTO {
  plan_code: string;
  plan_display_name: string;
  status: string;
  billing_cycle: string;
  provider: string;
  current_period_start: string;
  current_period_end: string;
  cancel_at_period_end: boolean;
  canceled_at: string | null;
}

export async function apiGetSubscription() {
  return apiJson<SubscriptionDTO>(`${API}/api/v1/billing/subscription`);
}

export interface UsageDTO {
  period_start: string;
  period_end: string;
  scan_limit: number | null;
  scans_used: number;
  scans_remaining: number | null;
  seat_limit: number | null;
  seats_used: number;
  breakdown: Record<string, number>;
}

export async function apiGetUsage() {
  return apiJson<UsageDTO>(`${API}/api/v1/billing/usage`);
}

export async function apiCancelSubscription() {
  return apiJson<SubscriptionDTO>(`${API}/api/v1/billing/subscription/cancel`, { method: "POST" });
}

export async function apiResumeSubscription() {
  return apiJson<SubscriptionDTO>(`${API}/api/v1/billing/subscription/resume`, { method: "POST" });
}

export async function apiChangePlan(plan_code: string, billing_cycle: "monthly" | "annual") {
  return apiJson<SubscriptionDTO>(`${API}/api/v1/billing/subscription/change`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan_code, billing_cycle }),
  });
}

export interface InvoiceDTO {
  id: string;
  amount_cents: number;
  currency: string;
  status: string;
  provider: string;
  hosted_url: string | null;
  pdf_url: string | null;
  period_start: string;
  period_end: string;
  issued_at: string;
  paid_at: string | null;
}

export async function apiListInvoices() {
  return apiJson<InvoiceDTO[]>(`${API}/api/v1/billing/invoices`);
}

// ── Payments — MTN MoMo ──

export async function apiMomoCheckout(
  plan_code: "clinician" | "practice",
  billing_cycle: "monthly" | "annual",
  phone: string,
) {
  return apiJson<{ intent_id: string; status: string; provider: string; poll_url: string }>(
    `${API}/api/v1/payments/momo/checkout`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plan_code, billing_cycle, phone }),
    },
  );
}

export interface PaymentIntentDTO {
  id: string;
  status: "requires_action" | "processing" | "succeeded" | "failed" | "canceled";
  provider: string;
  plan_code: string;
  billing_cycle: string;
  amount_cents: number;
  currency: string;
  confirmed_at: string | null;
}

export async function apiPollIntent(intentId: string) {
  return apiJson<PaymentIntentDTO>(`${API}/api/v1/payments/intents/${intentId}`);
}

// ── Orgs ──

export interface OrgDTO {
  id: string;
  name: string;
  slug: string;
  is_personal: boolean;
  is_active: boolean;
  role: string;
  created_at: string;
}

export interface MemberDTO {
  user_id: string;
  email: string;
  full_name: string | null;
  role: string;
  status: string;
  joined_at: string | null;
}

export interface InviteDTO {
  id: string;
  email: string;
  role: string;
  expires_at: string;
  created_at: string;
}

export async function apiListOrgs() {
  return apiJson<OrgDTO[]>(`${API}/api/v1/orgs`);
}

export async function apiListMembers(orgId: string) {
  return apiJson<MemberDTO[]>(`${API}/api/v1/orgs/${orgId}/members`);
}

export async function apiInviteMember(orgId: string, email: string, role: "admin" | "clinician" | "viewer") {
  return apiJson<InviteDTO>(`${API}/api/v1/orgs/${orgId}/invites`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, role }),
  });
}

export async function apiListInvites(orgId: string) {
  return apiJson<InviteDTO[]>(`${API}/api/v1/orgs/${orgId}/invites`);
}

export async function apiRevokeInvite(orgId: string, inviteId: string) {
  return apiJson<{ status: string }>(
    `${API}/api/v1/orgs/${orgId}/invites/${inviteId}/revoke`,
    { method: "POST" },
  );
}

export async function apiUpdateMemberRole(orgId: string, userId: string, role: "admin" | "clinician" | "viewer") {
  return apiJson<{ status: string; role: string }>(
    `${API}/api/v1/orgs/${orgId}/members/${userId}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role }),
    },
  );
}

export async function apiRemoveMember(orgId: string, userId: string) {
  return apiJson<{ status: string }>(`${API}/api/v1/orgs/${orgId}/members/${userId}`, {
    method: "DELETE",
  });
}

// ── Practice seats ──

export interface SeatStateDTO {
  included_seats: number;
  additional_seats: number;
  effective_limit: number | null;
  seats_used: number;
  can_buy_more: boolean;
  per_seat_cents: number;
  cycle: string;
}

export async function apiGetSeats() {
  return apiJson<SeatStateDTO>(`${API}/api/v1/billing/seats`);
}

export async function apiUpdateSeats(additional_seats: number) {
  return apiJson<SeatStateDTO>(`${API}/api/v1/billing/seats`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ additional_seats }),
  });
}

// ── Admin: webhook replay ──

export interface WebhookEventListItem {
  id: string;
  provider: "stripe" | "mtn" | "airtel" | "flutterwave" | "manual";
  provider_event_id: string;
  event_type: string | null;
  has_payload: boolean;
  received_at: string;
  processed_at: string | null;
  error: string | null;
}

export interface WebhookEventDetail extends WebhookEventListItem {
  payload: unknown;
}

export async function apiListWebhookEvents(filters: {
  provider?: string;
  state?: "ok" | "error" | "pending";
  limit?: number;
}) {
  const qs = new URLSearchParams();
  if (filters.provider) qs.set("provider", filters.provider);
  if (filters.state) qs.set("state", filters.state);
  if (filters.limit) qs.set("limit", String(filters.limit));
  return apiJson<WebhookEventListItem[]>(
    `${API}/api/v1/billing/admin/webhook-events${qs.toString() ? `?${qs}` : ""}`,
  );
}

export async function apiGetWebhookEvent(eventId: string) {
  return apiJson<WebhookEventDetail>(
    `${API}/api/v1/billing/admin/webhook-events/${eventId}`,
  );
}

export async function apiReplayWebhookEvent(eventId: string) {
  return apiJson<{ status: string; [k: string]: unknown }>(
    `${API}/api/v1/billing/admin/webhook-events/${eventId}/replay`,
    { method: "POST" },
  );
}
