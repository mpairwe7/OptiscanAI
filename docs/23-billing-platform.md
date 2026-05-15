# 23 · Subscription billing platform

This document covers the full SaaS layer added on top of the OptiscanAI screening
API: identity, tenancy, plan catalogue, quota enforcement, all four payment
rails, transactional email, in-product renewal banner, and Practice-tier seat
purchases.

It is the single source of truth that the marketing, backend, frontend,
deployment, and operations work flows feed into. If anything in the codebase
contradicts this page, *this page* is the bug.

---

## 1. Mental model

OptiscanAI ships in two configurations from the same codebase:

| Mode | Switches | Behaviour |
|---|---|---|
| **On-prem / research** | `DATABASE__ENABLED=false`, `BILLING__ENABLED=false`, `AUTH_ENABLED=false` | Single-tenant. No accounts. All endpoints public. Matches the original Phase-1–5 deployment story (Crane Cloud, HuggingFace Spaces). |
| **SaaS** | `DATABASE__ENABLED=true`, `BILLING__ENABLED=true`, `AUTH_ENABLED=true` | Multi-tenant. Users → Organizations → Subscriptions → Plans. Every metered or feature-gated endpoint requires a bearer JWT or `os_access` cookie. |

The feature gates and quota dependency *transparently no-op* when SaaS mode is
off, so existing on-prem deployments keep working without code changes.

---

## 2. Tier model

Four tiers are seeded in Alembic migration `0001`:

| Plan | Monthly | Annual (≈17 % off) | Scans / mo | Seats | Highlights |
|---|---|---|---|---|---|
| **Free** | $0 | $0 | 10 | 1 | Grad-CAM only, 7-day report retention, watermarked PDFs |
| **Clinician** | $29 | $290 | 500 | 1 | All XAI (Grad-CAM, LIME, SHAP, IG, ELI5), clinical reasoning, voice mode, 1-year retention |
| **Practice** | $149 | $1 490 | 3 000 | 5 (extras available) | Multi-seat review queue, audit log, fairness dashboard |
| **Health System** | contact-sales | contact-sales | unlimited | unlimited | SSO/SCIM, BAA, DHIS2 + FHIR + DICOM + SMS, dedicated CSM |

Tier order is `free < clinician < practice < health_system`. Feature gates use
this rank for `require_tier(min_tier=…)` checks.

Frontend mirror: `frontend/src/lib/plans.ts` (used at build/SSR; runtime values
come from `GET /api/v1/billing/plans` which reads the DB).

---

## 3. Authentication

Source files: `backend/app/core/auth.py`, `backend/app/core/security.py`,
`backend/app/services/auth_service.py`, `backend/app/routers/auth.py`.

### Token strategy
- **Access JWT** — 15-min TTL, signed HS256, claims `sub` (user_id), `org`
  (active organization id), `role`, `exp`, `typ=access`. Stored as `os_access`
  httpOnly cookie (`SameSite=Lax`, `Secure` in production, path `/`).
- **Refresh token** — 30-day rotating opaque token. SHA-256-hashed at rest in
  `refresh_tokens`. Stored as `os_refresh` httpOnly cookie. Every `/refresh`
  call revokes the old row and issues a fresh one.
- **Anonymous fallback** — when `AUTH_ENABLED=false`, `get_current_user` returns
  a synthetic `TokenPayload(sub="anonymous", role="admin")` so legacy callers
  keep working.

### Endpoints
| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Creates user + personal Organization + Free Subscription. Sends verification email. |
| `POST` | `/api/v1/auth/login` | Email + password → token pair. |
| `POST` | `/api/v1/auth/refresh` | Rotates refresh; issues new access. |
| `POST` | `/api/v1/auth/logout` | Revokes refresh row and clears cookies. |
| `POST` | `/api/v1/auth/magic-link/request` | Sends a one-shot signed link. Silent — does not reveal account existence. |
| `GET` | `/api/v1/auth/magic-link/verify?token=…` | Exchanges the link for a session. Auto-creates the user on first sign-in. |
| `GET` | `/api/v1/auth/verify-email?token=…` | Confirms ownership of the email. |
| `POST` | `/api/v1/auth/verify-email/resend` | Re-issues the verification mail. |
| `POST` | `/api/v1/auth/password/forgot` | Issues a reset link. Silent. |
| `POST` | `/api/v1/auth/password/reset` | Completes the reset. |
| `GET` | `/api/v1/auth/me` | Returns the rich auth context — user, org, role, subscription summary. |

### Front-door layout
Public marketing routes live under `(public)/` (landing, `/pricing`, `/sign-in`,
`/sign-up`, `/reset-password`, `/verify-email`, `/contact-sales`, `/legal/*`).

Authenticated routes live under `(app)/app/*` and are gated by `proxy.ts` —
a request without either cookie is redirected to `/sign-in?next=…`.

---

## 4. Quota & feature gates

### Scan quota
`backend/app/core/quota.py::check_scan_quota_inline` is called inside
`POST /api/v1/predict` *before* model inference:

1. Resolves the auth context (header `Authorization: Bearer …` or `os_access` cookie).
2. Looks up the org's active Subscription + Plan.
3. Counts `UsageEvent` rows with `event_type=scan` between
   `subscription.current_period_start` and `current_period_end` (composite
   index `(organization_id, event_type, occurred_at)` makes this O(log n)).
4. If `count >= plan.scan_limit_monthly` and the limit is not `None`, raises
   `HTTPException(402)` with payload:
   ```json
   {
     "error": "quota_exceeded",
     "plan": {"code": "free", "scan_limit_monthly": 10},
     "usage": {"used": 10, "limit": 10, "resets_at": "2026-06-01T00:00:00Z"},
     "upgrade_url": "/pricing",
     "recommended_plan": "clinician"
   }
   ```
5. Adds `X-Usage-Used`, `X-Usage-Limit`, `X-Usage-Resets` headers to every
   successful response.

On the frontend, `apiFetch` (in `frontend/src/lib/api-fetch.ts`) catches 402s
with `quota_exceeded` and opens the `PaywallModal` automatically.

### Feature gates
`backend/app/core/feature_gate.py::require_tier("min_tier", feature="…")`
returns a FastAPI dependency that resolves to the auth context if the user's
plan rank is high enough, or raises 403 with payload:
```json
{
  "error": "feature_locked",
  "feature": "shap",
  "required_plan": "clinician",
  "current_plan": "free",
  "upgrade_url": "/pricing"
}
```

The frontend interceptor opens `UpsellSheet` on this shape.

Current gate map (all auto-disabled when `BILLING__ENABLED=false`):

| Router | Min tier |
|---|---|
| `/api/v1/predict`, `/api/v1/explain/gradcam`, `/api/v1/diseases`, `/api/v1/health/*` | Free |
| `/api/v1/explain/{lime,shap,integrated-gradients,eli5,comprehensive}` | Clinician |
| `/api/v1/clinical/*`, `/api/v1/agents/*`, `/api/v1/voice/*` | Clinician |
| `/api/v1/review/*`, `/api/v1/governance/*`, `/api/v1/admin/*` (monitoring) | Practice |
| `/api/v1/dhis2/*`, `/api/v1/fhir/*`, `/api/v1/dicom/*`, `/api/v1/sms/send-referral`, `/api/v1/sms/delivery/*` | Health System |

SMS provider callbacks (`/api/v1/sms/callback`, `/api/v1/sms/ussd`) remain
**public** — they are provider-signed webhooks from Africa's Talking.

---

## 5. Payment rails

### Stripe (cards)
Source: `backend/app/integrations/stripe_client.py`, `backend/app/routers/payments.py`.

**Checkout flow**
1. Frontend `POST /api/v1/payments/stripe/checkout-session` with `{plan_code, billing_cycle}`.
2. Backend creates a `subscription` mode Checkout Session with metadata
   (`organization_id`, `plan_code`, `billing_cycle`, `user_id`) and a fresh
   idempotency key.
3. Frontend redirects browser to `session.url` (Stripe-hosted page).
4. User completes payment.
5. Stripe redirects to `STRIPE__SUCCESS_URL` (`/app/checkout/success?session_id={CHECKOUT_SESSION_ID}`).
6. **Independently**, Stripe POSTs to `/api/v1/payments/stripe/webhook`.
7. Webhook handler verifies `stripe-signature`, idempotency-checks via
   `webhook_events`, and reconciles via `apply_stripe_subscription_event` →
   upserts Subscription, syncs status/period, syncs `additional_seats` from
   the subscription items, flips `Organization.plan_id`, busts the quota cache.
8. Frontend `CheckoutSuccess` polls `/api/v1/auth/me` for up to 30 s waiting
   for `subscription.plan.code !== "free"`.

**Events handled**
- `checkout.session.completed`
- `customer.subscription.{created,updated,deleted}` — also syncs seat quantity
- `invoice.paid` — writes Invoice row
- `invoice.payment_failed` — flips Subscription to `past_due`

### MTN MoMo + Airtel Money
Source: `backend/app/services/momo_billing_service.py`,
`backend/app/integrations/mobile_money/client.py`.

MoMo doesn't support recurring auto-debit, so each cycle is a **one-shot push
prompt**:

1. `POST /api/v1/payments/momo/checkout` with `{plan_code, billing_cycle, phone, provider}`.
2. Backend converts plan USD → UGX using `MOBILE_MONEY__UGX_PER_USD` (default
   3 800), creates a `PaymentIntent`, asks `MobileMoneyClient.request_payment`
   to push a USSD prompt to the user's phone.
3. Backend responds `{intent_id, status, poll_url}`.
4. Frontend shows an "Awaiting payment confirmation" screen and polls
   `GET /api/v1/payments/intents/{intent_id}` every 3 s.
5. User enters MoMo PIN on the phone.
6. MTN/Airtel sends a callback to `/api/v1/payments/momo/callback/{mtn,airtel}`.
7. Callback handler verifies HMAC-SHA256 against `MOBILE_MONEY__MTN_CALLBACK_SECRET`
   (or `AIRTEL_CALLBACK_SECRET`), idempotency-checks via `webhook_events`,
   confirms the intent → advances Subscription `current_period_end` by 30 or 365
   days, writes Invoice.

When the user's phone never confirms, the polling endpoint queries the provider
directly via `MobileMoneyClient.check_payment_status`.

### Flutterwave
Source: `backend/app/integrations/flutterwave/client.py`,
`backend/app/services/momo_billing_service.py::initiate_flutterwave_payment`.

Flutterwave is a hosted redirect (no PIN push):

1. `POST /api/v1/payments/flutterwave/checkout`.
2. Backend creates a `PaymentIntent` with a `tx_ref`, calls Flutterwave
   `POST /v3/payments`, gets a hosted-checkout `link`.
3. Frontend redirects browser to `link`.
4. User completes payment on Flutterwave's page.
5. Flutterwave POSTs to `/api/v1/payments/flutterwave/webhook` with a
   `verif-hash` header.
6. Webhook handler verifies the hash (constant-time compare against
   `FLUTTERWAVE__SECRET_HASH`), **double-checks** by calling
   `GET /v3/transactions/verify_by_reference`, and only then confirms the
   intent.

### Idempotency
All four rails funnel into `webhook_events` (unique on
`(provider, provider_event_id)`). Replaying any callback is a no-op.

### Payment-provider configuration table

| Variable | Required when | Purpose |
|---|---|---|
| `STRIPE__ENABLED=true` | Stripe used | Master switch |
| `STRIPE__API_KEY=sk_...` | Stripe used | Secret API key |
| `STRIPE__WEBHOOK_SECRET=whsec_...` | Stripe used | Verifies webhook bodies |
| `STRIPE__CLINICIAN_MONTHLY_PRICE_ID=price_...` | Clinician plan | Stripe Price |
| `STRIPE__CLINICIAN_ANNUAL_PRICE_ID=price_...` | Clinician annual | Stripe Price |
| `STRIPE__PRACTICE_MONTHLY_PRICE_ID=price_...` | Practice plan | Stripe Price |
| `STRIPE__PRACTICE_ANNUAL_PRICE_ID=price_...` | Practice annual | Stripe Price |
| `STRIPE__PRACTICE_EXTRA_SEAT_MONTHLY_PRICE_ID=price_...` | Practice extra seats | Per-seat Stripe Price |
| `STRIPE__PRACTICE_EXTRA_SEAT_ANNUAL_PRICE_ID=price_...` | Practice annual extras | Per-seat Stripe Price |
| `MOBILE_MONEY__ENABLED=true` | MoMo used | Master switch |
| `MOBILE_MONEY__MTN_*` | MTN MoMo | API/secrets (see config.py) |
| `MOBILE_MONEY__AIRTEL_*` | Airtel | API/secrets |
| `MOBILE_MONEY__UGX_PER_USD=3800` | MoMo used | FX rate, plan USD → UGX charge |
| `FLUTTERWAVE__ENABLED=true` | Flutterwave used | Master switch |
| `FLUTTERWAVE__SECRET_KEY=FLWSECK_...` | Flutterwave used | API |
| `FLUTTERWAVE__SECRET_HASH=...` | Flutterwave used | Webhook auth |

---

## 6. Transactional email

Source: `backend/app/services/email_service.py`,
`backend/app/services/email_templates.py`.

### Provider switch
`EMAIL__PROVIDER` ∈ `console | smtp | resend | sendgrid`. `console` (default)
logs the email body to stdout — useful for dev and CI. Production should use
Resend or SendGrid.

### Retry behaviour
HTTP providers (Resend, SendGrid) retry once with linear backoff on 5xx or
network errors. 4xx is logged but not retried (it's a bug in our payload).
`send_email` **never raises** — a Resend outage will not 500 registration.

### Templates
Pure-Python factories return `RenderedEmail(subject, body_text, body_html)`.
Both variants always populated; HTML uses inline styles since most mail
clients strip `<style>`. Each template includes the OptiscanAI brand bar,
honorific-aware greeting (`Dr Jane Doe` → "Hi Jane,"), and footer with support
contact and the MakStartup physical address.

| Template | Subject | Trigger |
|---|---|---|
| `email_verification` | "Verify your OptiscanAI account" | `POST /auth/register`, `POST /auth/verify-email/resend` |
| `magic_link` | "Sign in to OptiscanAI" | `POST /auth/magic-link/request` |
| `password_reset` | "Reset your OptiscanAI password" | `POST /auth/password/forgot` |
| `org_invite` | "You're invited to {org} on OptiscanAI" | `POST /orgs/{id}/invites` |
| `renewal_reminder` | "Renew your {plan} plan ({n} days left)" | Renewal cron |

### Provider env vars
```bash
EMAIL__ENABLED=true
EMAIL__PROVIDER=resend
EMAIL__FROM_ADDRESS=noreply@makstartup.com
EMAIL__FROM_NAME=OptiscanAI
EMAIL__RESEND_API_KEY=re_...
# or
EMAIL__SENDGRID_API_KEY=SG.xxx
# or for SMTP
EMAIL__SMTP_HOST=smtp.example.com
EMAIL__SMTP_PORT=587
EMAIL__SMTP_USERNAME=…
EMAIL__SMTP_PASSWORD=…
```

DNS prerequisites: set SPF, DKIM, and DMARC records on `makstartup.com` per
the provider's onboarding so outbound mail doesn't land in spam.

---

## 7. Renewal-reminder cron

Source: `backend/app/services/renewal_service.py`,
`backend/app/cli/renewal_reminders.py`,
`backend/app/models/renewal_reminder.py`.

### What it does
Each run picks Subscriptions where:
- `provider ∈ (mtn, airtel, flutterwave)`
- `status ∈ (active, trialing)`
- `cancel_at_period_end = false`
- `current_period_end` between `now − 1 day` and `now + 7 days`

Maps each to a `ReminderKind` based on days-to-expiry (`7d`, `3d`, `1d`,
`expired`). For each `(subscription_id, period_end, kind)` triple that
doesn't already have a non-error row in `renewal_reminders`, sends the
`renewal_reminder` email and inserts a row.

The triple unique constraint guarantees the cron is idempotent — re-running
the same day is a no-op until the subscription's `period_end` advances after a
paid renewal (which automatically opens a fresh set of reminder slots).

### Triggering
External cron:
```cron
# /etc/cron.d/optiscan-renewals
0 2 * * * uvuser cd /srv/optiscan && uv run python -m backend.app.cli.renewal_reminders >> /var/log/optiscan/renewals.log 2>&1
```

The script emits one-line JSON to stdout:
```json
{"event":"renewal_reminders","found_subscriptions":42,"reminders_sent":18,"reminders_skipped":24,"errors":0}
```

Manual trigger (superuser only):
```bash
curl -X POST https://www.optiscan.makstartup.com/api/v1/billing/admin/run-renewal-reminders \
     -H "Authorization: Bearer $ACCESS"
```

### In-product surface
`frontend/src/components/billing/renewal-banner.tsx` reads the same
`/api/v1/billing/subscription` data and shows a tone-coded banner (blue / amber
/ red as expiry approaches) at the top of every `/app/*` page. A second
in-card variant lives on `/app/billing`. The global variant is dismissible
per `period_end` via `localStorage`.

---

## 8. Practice seats (Phase F)

Source: `backend/app/services/seat_service.py`, the `/api/v1/billing/seats`
endpoints, `frontend/src/components/team/seat-manager.tsx`.

### Model
The Practice plan ships with `seat_limit = 5`. Extra seats live on the
Subscription:

```
effective_seat_limit = plan.seat_limit + subscription.additional_seats
```

On Stripe, the extra-seat add-on is a **separate Subscription Item** with its
own Price. `subscription.stripe_seat_item_id` caches the item id so subsequent
quantity updates hit the same line.

### Buying / removing seats
`POST /api/v1/billing/seats {additional_seats: int}` (owners/admins only):
- Only available when subscription is on Stripe and plan is Practice
- Rejects values that would drop the effective limit below `seats_used`
- Calls `stripe_client.set_seat_quantity` — creates, modifies, or deletes the
  seat Subscription Item with `proration_behavior="create_prorations"`
- Stripe immediately charges (or credits) the prorated amount
- Webhook re-syncs `additional_seats` from the next
  `customer.subscription.updated` event

### Invite enforcement
`org_service.invite_member` now uses the effective limit; the 402 payload
returns both `included_seats` and `additional_seats` plus an
`upgrade_url: /app/team` (rather than `/pricing`) so the UI nudges seat
purchases instead of plan upgrades.

### UI
`/app/team` mounts the `<SeatManager />` component above the invite form. It
shows current usage, lets the user step up/down via +/− buttons with a numeric
input, and confirms the prorated charge before calling the backend. Health
System users see a "Unlimited seats" panel; non-Stripe Practice users see a
"contact sales" panel because MoMo can't pro-rate seat purchases.

---

## 9. Database & migrations

| Revision | Title | What it adds |
|---|---|---|
| `0001` | Initial billing schema | All core tables + 4-plan seed |
| `0002` | renewal_reminders | The reminder log table |
| `0003` | additional_seats | `subscriptions.additional_seats` (int, default 0) + `subscriptions.stripe_seat_item_id` |

Apply:
```bash
uv run alembic upgrade head
```

Rollback the most recent:
```bash
uv run alembic downgrade -1
```

### Tables
- `users` · `organizations` · `memberships` · `plans` · `subscriptions` ·
  `invoices` · `payment_intents` · `usage_events` · `webhook_events` ·
  `refresh_tokens` · `email_verification_tokens` · `password_reset_tokens` ·
  `magic_link_tokens` · `organization_invites` · `renewal_reminders`

Primary keys: UUIDv4 for user-facing tables, `bigint` for `usage_events`
(high write volume). Hot-path composite index
`(organization_id, event_type, occurred_at)` makes the monthly quota query
O(log n).

---

## 10. Runbook

### First-time bring-up
```bash
# 1. Install Python deps (sqlalchemy, asyncpg, alembic, argon2-cffi, pyjwt, stripe, httpx)
uv sync

# 2. Bring up Postgres (any 14+ works)
docker run -d --name optiscan-pg -e POSTGRES_USER=optiscan -e POSTGRES_PASSWORD=optiscan \
  -e POSTGRES_DB=optiscan -p 5432:5432 postgres:16

# 3. Apply migrations + seed plans
export DATABASE__ENABLED=true
export DATABASE__URL=postgresql+asyncpg://optiscan:optiscan@localhost:5432/optiscan
uv run alembic upgrade head

# 4. Start backend in SaaS mode
export AUTH_ENABLED=true
export BILLING__ENABLED=true
export JWT_SECRET=$(openssl rand -hex 32)
export PUBLIC_APP_URL=https://www.optiscan.makstartup.com
uv run uvicorn backend.app.main:app --port 8080

# 5. Start frontend
cd frontend && BACKEND_URL=http://localhost:8080 npm run dev
# → http://localhost:3000
```

### Daily ops
- **Renewal reminders**: cron entry above; check `/var/log/optiscan/renewals.log`
- **Webhook health**: `SELECT provider, count(*), max(received_at) FROM webhook_events GROUP BY provider;`
- **Drift in subscription state vs Stripe**: trigger
  `POST /api/v1/payments/stripe/webhook` replay from the Stripe dashboard
  for any suspicious customer

### Promoting a superuser
```sql
UPDATE users SET is_superuser = true WHERE email = 'ops@makstartup.com';
```
Required to hit the admin endpoints (run-renewal-reminders, future seat
overrides, etc.).

### Provisioning Stripe
1. Create one Product per plan: "Clinician", "Practice".
2. Create two Prices per Product: monthly and annual.
3. Create a third Product "Practice extra seat" + monthly and annual Prices.
4. Copy the `price_…` IDs into the matching `STRIPE__*_PRICE_ID` env vars.
5. Create a webhook endpoint pointing at
   `https://www.optiscan.makstartup.com/api/v1/payments/stripe/webhook` and
   listen for: `checkout.session.completed`,
   `customer.subscription.created|updated|deleted`, `invoice.paid`,
   `invoice.payment_failed`. Copy the signing secret to `STRIPE__WEBHOOK_SECRET`.

### Provisioning MTN / Airtel
1. Register an organisation account at https://momodeveloper.mtn.com (or the
   Airtel developer portal).
2. Create a Collections subscription product; copy `Ocp-Apim-Subscription-Key`
   into `MOBILE_MONEY__MTN_SUBSCRIPTION_KEY`.
3. Generate `apiKey` + `apiUser`; set `MOBILE_MONEY__MTN_API_KEY/_SECRET`.
4. In the MTN portal, set the callback URL to
   `https://www.optiscan.makstartup.com/api/v1/payments/momo/callback/mtn`
   and shared secret to `MOBILE_MONEY__MTN_CALLBACK_SECRET`.
5. Repeat for Airtel.

### Provisioning Flutterwave
1. Create a Flutterwave account, copy `secret_key` and `public_key`.
2. In dashboard → Webhooks set URL to
   `https://www.optiscan.makstartup.com/api/v1/payments/flutterwave/webhook`
   and copy the "Secret hash" → `FLUTTERWAVE__SECRET_HASH`.

### Provisioning email
1. Add `makstartup.com` to Resend (or SendGrid).
2. Set SPF: `v=spf1 include:_spf.resend.com -all`
3. Set DKIM TXT record per provider instructions.
4. Set DMARC: `v=DMARC1; p=quarantine; rua=mailto:dmarc@makstartup.com`
5. Set `EMAIL__PROVIDER=resend`, `EMAIL__RESEND_API_KEY=…`,
   `EMAIL__FROM_ADDRESS=noreply@makstartup.com`.

---

## 11. Local QA checklist (end-to-end)

Cold start with `BILLING__ENABLED=true`:

1. `GET /` → marketing landing (no cookies). Pricing, FAQ, Hero render. ✓
2. `POST /api/v1/auth/register` with a new email → 201, sets `os_access` + `os_refresh` cookies, sends verification email (visible in `console` log). ✓
3. `GET /api/v1/auth/me` → user + personal org + Free subscription. ✓
4. `POST /api/v1/predict` × 10 with a fundus image → all succeed. ✓
5. `POST /api/v1/predict` × 11 → 402 with `quota_exceeded`. Frontend opens `PaywallModal`. ✓
6. `POST /api/v1/explain/shap` on Free → 403 with `feature_locked`. Frontend opens `UpsellSheet`. ✓
7. Subscribe to Clinician via Stripe test card → webhook fires, `/api/v1/auth/me` flips to clinician, quota chip shows `0/500`. ✓
8. Upgrade to Practice, invite 4 members from `/app/team`, hit seat limit on the 6th invite → 402 with `seat_limit_reached` and link to `/app/team`. ✓
9. `POST /api/v1/billing/seats {additional_seats: 3}` → Stripe pro-rates, effective limit becomes 8. Invite the 6th member again — succeeds. ✓
10. Use `/api/v1/payments/momo/checkout` (MTN sandbox) → push prompt, callback → subscription advances 30 days, invoice row appears. ✓
11. Replay the same MTN callback → 200 `{status: duplicate}`. No double-invoice. ✓
12. Run `uv run python -m backend.app.cli.renewal_reminders` against a subscription with `period_end = now + 5d` → one `7d`-kind reminder row added, email printed. ✓
13. Run again immediately → 0 sent, 1 skipped (idempotent). ✓
14. With a MoMo subscription within 7 days of `current_period_end`, log in → global `RenewalBanner` appears at the top of every `/app/*` page; click "Renew now" → routes to `/app/checkout/<plan>`. ✓
15. As `is_superuser`, visit `/app/admin/webhooks` → table renders recent events. Click a Stripe `customer.subscription.updated` row → drawer shows full payload. Click "Replay event" → returns `{status: replayed, subscription_id: …}`. ✓
16. On `/app/billing` for a Clinician sub, click **Downgrade to Free** → confirm modal shows Stripe-prorate language; confirm → Stripe cancels with prorated credit, account flips to Free. ✓
17. With 4 active Practice members, hit Downgrade to Free → 400 `members_exceed_target_seats` with the active-member count; remove 3 members, retry → success. ✓

---

## 12. Where the bodies are buried

A few non-obvious decisions worth knowing:

- **The Stripe extra-seat item is created lazily** — a fresh Practice
  subscription has zero seat items until the first `POST /seats` call. This
  keeps webhook reconciliation simple (no special "0 quantity" item).
- **MoMo subscriptions don't auto-renew.** This is an inherent constraint of
  the rails, not a bug. The renewal cron + in-product banner cover this.
- **Free tier "renews" via the agent orchestrator nightly job** — see
  `billing_service.roll_period_forward_if_needed`. Stripe webhooks bump period
  dates for paid subs; MANUAL provider gets bumped here.
- **`AUTH_ENABLED=false` is the kill switch** — when disabled, every
  `require_tier` dependency short-circuits to `None` at app start, so on-prem
  deployments incur zero runtime cost from the gates.
- **The marketing-page `BACKEND_URL` is set on the Next runtime, not the
  browser.** Browser requests hit `/api/v1/*` on the Next origin and are
  rewritten server-side (`next.config.ts`). This keeps auth cookies first-party.

---

## 13. Future work

| Item | Priority | Effort |
|---|---|---|
| Annual-to-monthly switch mid-cycle | low | small |
| Email-preference centre (opt-out of renewal reminders) | low | small |
| Flutterwave hosted MoMo with auto-renew via tokenisation | medium | medium |
| Africa's Talking SMS renewal reminder (in addition to email) | low | small |
| Datadog / Better Stack hookup for cron-failure paging | medium | small |
| Kustomize overlay + sealed-secrets for the Postgres Secret | medium | small |
| Alembic-as-a-Job (pre-deploy hook) instead of running in the backend pod | low | small |
| wal-g PITR backups for the Postgres StatefulSet | medium | medium |

---

## 14. Database hosting — current and upgrade paths

The billing layer needs Postgres. There are three deployment shapes; this
section covers the one we ship today (Option 1) and the two future upgrade
paths (Options 2 and 3).

### Option 1 — Embedded Postgres in the application container *(fallback / HF Spaces)*

Used when a sidecar isn't available — Hugging Face Spaces, standalone
`docker run` without compose, single-pod demos. Active when
`EMBEDDED_POSTGRES__ENABLED=true` (default if unset). compose / k8s set it to
`false` so the in-image cluster sleeps and the app talks to the sidecar.

Both `Dockerfile` and `Dockerfile.cpu` install
`postgresql-14` alongside the app. Supervisord runs four programs at startup
(priority order — lower starts first):

| Priority | Program | Role |
|---|---|---|
| 10 | `postgres` | `postgres-bootstrap.sh` initialises the cluster on first run, then `exec`s the postgres process |
| 20 | `backend` | `backend-start.sh` waits for Postgres readiness, runs `alembic upgrade head`, then `exec`s uvicorn |
| 30 | `frontend` | Next.js standalone |
| 40 | `nginx` | Reverse proxy on `:8080` |

**Bootstrap behaviour** (`scripts/container/postgres-bootstrap.sh`):
- Detects empty PGDATA → `initdb` with `scram-sha-256`, creates `optiscan`
  role with the password from `POSTGRES_PASSWORD` env (default `optiscan`),
  creates the `optiscan` database
- Configures `listen_addresses = '127.0.0.1'` only — Postgres is reachable
  exclusively from within the container
- Subsequent boots skip init and just start postgres
- Process runs as the `postgres` system user

**Wait + migrate** (`scripts/container/backend-start.sh`):
- When `DATABASE__ENABLED=true`, polls `pg_isready` for up to 60 s
- Runs `alembic upgrade head` (idempotent — only missing migrations apply)
- Backend never crashes if Postgres is slow — degrades to "billing endpoints
  503" instead

**Persistence**
- `docker-compose.yml` mounts the named volume `optiscan-pgdata` (or
  `optiscan-pgdata-cpu`) at `/var/lib/postgresql/data`
- Survives `docker compose down`; cleared by `docker compose down -v`
- On Crane Cloud the equivalent is a `PersistentVolumeClaim` mounted at the
  same path — without one, every pod restart resets the database

**Trade-offs**
- ✅ Single container — fewer moving parts, fits HF Spaces / single-pod
  K8s deployments
- ✅ Same image works for on-prem and SaaS — just flip `DATABASE__ENABLED`
- ⚠️ Postgres + uvicorn + Next.js + nginx share the same memory budget
  (~150 MB postgres baseline)
- ⚠️ Restart restarts the database — brief read/write outage on every deploy
- ❌ Not horizontally scalable — the app is stateful

**Suitable for**: development, single-pod pilots, demos. **Move on once you
hit ~100 paying organisations or need zero-downtime deploys.**

### Option 2 — Sidecar Postgres container *(current default for compose and k8s)*

The shipping default for any compose- or Kubernetes-based deploy. The
in-container Postgres still exists in the image but `exec sleep infinity`s
out when `EMBEDDED_POSTGRES__ENABLED=false`, which the api services set.

**What's wired today**

`docker-compose.yml` defines a `postgres` service running `postgres:16-alpine`:
- env-driven credentials (`POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB`)
- TCP healthcheck via `pg_isready`
- Persistent `optiscan-pgdata` named volume mounted at
  `/var/lib/postgresql/data`
- Loopback port binding `127.0.0.1:5432:5432` so a developer can `psql`
  directly without exposing the DB to the LAN
- Both `api` and `api-cpu` services declare
  `depends_on: postgres: condition: service_healthy`, so `docker compose up`
  only starts the app once Postgres reports ready
- `DATABASE__URL=postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}`
  composed at compose time and injected into the api containers
- `EMBEDDED_POSTGRES__ENABLED=false` set on both api services

`k8s/base/` ships three Postgres manifests:
- `postgres-secret.yaml` — `Opaque` Secret with `POSTGRES_USER` /
  `POSTGRES_PASSWORD` / `POSTGRES_DB`. Rotate via
  `kubectl create secret --dry-run=client -o yaml | kubectl apply -f -`.
- `postgres-service.yaml` — headless Service so backend pods reach
  `optiscan-postgres.retinalai.svc.cluster.local`.
- `postgres-statefulset.yaml` — single-replica StatefulSet, 20 Gi
  `PersistentVolumeClaim` per pod, `pg_isready` readiness + liveness,
  `fsGroup: 999` (postgres group inside `postgres:16-alpine`), requests
  `100m CPU / 256 Mi memory`, limits `1000m / 1 Gi`.
- `backend-deployment.yaml` references both — the Secret feeds the env vars
  that build `DATABASE__URL` against the headless Service.

**Bring it up — compose**

```bash
# In .env: POSTGRES_PASSWORD=$(openssl rand -hex 32)
#          JWT_SECRET=$(openssl rand -hex 32)
docker compose up -d                            # postgres + api
docker compose logs -f postgres api             # watch alembic run on first start
docker compose exec postgres psql -U optiscan -d optiscan -c '\dt'
```

**Bring it up — Kubernetes**

```bash
# Replace POSTGRES_PASSWORD in postgres-secret.yaml before applying
kubectl apply -f k8s/base/postgres-secret.yaml
kubectl apply -f k8s/base/postgres-service.yaml
kubectl apply -f k8s/base/postgres-statefulset.yaml
kubectl rollout status statefulset/optiscan-postgres -n retinalai
kubectl apply -f k8s/base/backend-deployment.yaml
```

**Migration from Option 1 to Option 2** (only relevant if you started on
embedded and need to preserve data):

1. `docker compose exec api su postgres -c "/usr/lib/postgresql/14/bin/pg_dump -U optiscan optiscan"` → dump
2. `docker compose up -d postgres`
3. `docker compose exec -T postgres psql -U optiscan optiscan < dump.sql`
4. Set `EMBEDDED_POSTGRES__ENABLED=false` (done automatically by current compose)
5. Restart the api: `docker compose restart api`

**Trade-offs**
- ✅ App stateless — horizontal scale works
- ✅ Postgres restart doesn't restart the app
- ✅ Can run a managed Postgres image (`postgres:16-alpine`, `timescale/timescaledb`)
- ⚠️ One more thing to back up
- ⚠️ Still self-hosted — you own the upgrade path, the WAL archiving, the PITR

**Suitable for**: production deployments up to ~10 000 organisations or until
you outgrow a single Postgres node.

### Option 3 — Managed Postgres *(recommended at production scale)*

The same architecture as Option 2, but the Postgres node is operated by a
managed-service provider. App pods point `DATABASE__URL` at the managed
endpoint and the sidecar disappears entirely.

**Provider options** (in order of fit for OptiscanAI):

1. **Crane Cloud Managed Postgres** — same region as the app pods, billed
   per-GB-month. No cross-border data transfer (Uganda PDP friendly).
2. **AWS RDS for Postgres** — Multi-AZ, point-in-time-recovery, IAM auth,
   read replicas. Cross-border by default (Frankfurt / Ireland for African
   workloads). Pair with an AWS-region close to Uganda (`af-south-1` in Cape
   Town) for data-residency.
3. **Supabase / Neon** — Postgres-as-a-Service with HTTP client SDKs. Useful
   if you want low-ops + branching for staging environments.

**Configuration** (single env-var change):

```bash
# Replace the in-cluster URL
DATABASE__URL=postgresql+asyncpg://optiscan:${PG_PASSWORD}@db-host.cranecloud.io:5432/optiscan?ssl=require
```

The `?ssl=require` query parameter is critical — managed providers expose
Postgres over TLS only. SQLAlchemy + asyncpg honours that flag.

**Migration from Option 2 to Option 3:**
1. Snapshot the sidecar (`pg_dump`)
2. Provision the managed instance, restore the dump
3. Update `DATABASE__URL` secret
4. Roll a deploy; old sidecar drains and can be torn down
5. (Optional) Enable RDS Proxy / Postgres-flavoured connection pooler if
   the per-pod pool footprint is too high

**Trade-offs**
- ✅ Backups, PITR, failover, encryption-at-rest are operator concerns
- ✅ Scales independently of the app
- ✅ Compliance audits get easier (SOC 2, HIPAA, PDP) — provider attestations
  cover the database tier
- ⚠️ Cost: $50-300/month minimum on most providers vs. a few cents of
  compute for the sidecar
- ⚠️ Network egress charges when cross-region

**Suitable for**: production at any scale where the engineering hours saved
on backup-tooling and PITR exceed the managed-service premium.

### Quick reference — which option to pick

| You are... | Pick |
|---|---|
| **docker-compose / k8s deploy (most cases)** | **Option 2 — already wired** |
| Hugging Face Spaces / standalone `docker run` (no sidecar possible) | Option 1 |
| Approaching 1 000+ paying orgs, need PITR + SOC 2 attestation | Option 3 |
| Subject to a Ugandan data-residency clause | Crane Cloud Managed Postgres (Option 3) |

---

## 15. In-product UX surfaces

The billing layer surfaces in the web app via these components, all
auto-disabled when `BILLING__ENABLED=false`:

| Component | File | Where it shows | What it does |
|---|---|---|---|
| `PaywallModal` | `frontend/src/components/billing/paywall-modal.tsx` | Global (mounted in `providers.tsx`) | Opens automatically when `apiFetch` sees a 402 with `quota_exceeded`. Shows usage / limit, live countdown to `resets_at`, primary "Upgrade to {plan}" CTA. |
| `UpsellSheet` | `frontend/src/components/billing/upsell-sheet.tsx` | Global | Opens on 403 with `feature_locked`. Pre-renders the cheapest tier that unlocks the requested feature; secondary "Compare all plans" link. |
| `UsageChip` | `frontend/src/components/billing/usage-chip.tsx` | Sidebar + `MobileTopBar` | Polls `/api/v1/billing/usage` every 30 s. Teal &lt;80 %, amber 80–99 %, red ≥100 %. Click → `/app/usage`. |
| `RenewalBanner` (global) | `frontend/src/components/billing/renewal-banner.tsx` | `(app)/layout.tsx` — every authed page | Shows when a MoMo/Flutterwave subscription is within 7 days of expiry. Sky-blue → amber (≤3 d) → red (≤1 d). Dismissible per `period_end` via `localStorage`. |
| `RenewalBanner` (inline) | same file, `variant="inline"` | `/app/billing` | Always visible — not dismissible on the billing surface. |
| `SeatManager` | `frontend/src/components/team/seat-manager.tsx` | `/app/team` | +/− counter for Practice extra seats. Shows prorated cost preview and effective new limit. Two-step commit. |
| `DowngradeDialog` | `frontend/src/components/billing/downgrade-dialog.tsx` | `/app/billing` | Confirmation modal for the Free-tier downgrade. Surfaces Stripe-prorate language for card subs, "current period forfeit" for MoMo. |
| `PricingCard` + `PricingCards` | `frontend/src/components/billing/{pricing-card,billing-period-toggle}.tsx`, `(public)/pricing/pricing-cards.tsx` | `/pricing` + marketing teaser | 4-card pricing grid with monthly/annual toggle (~17 % off). Used inline by paywall + upsell too. |

The **marketing site** itself is composed of independent server components
under `frontend/src/components/marketing/`:
`hero` (with the SVG `fundus-mockup`), `trust-strip`, `logo-cloud`,
`how-it-works`, `value-props`, `compliance-badges`, `testimonials`,
`pricing-teaser`, `faq`, `marketing-nav`, `marketing-footer`. The landing
page composes them in `src/app/(public)/page.tsx`. Add a new section by
appending a new component there — no client-side wiring required.

Legal pages live at `/legal/privacy` and `/legal/terms`, rendered through
the shared `LegalDoc` component (`frontend/src/components/marketing/legal-doc.tsx`).
Add a clause by appending a `LegalSection` entry to the page's `SECTIONS`
array — the table of contents and anchors update automatically.

---

## 16. Admin / ops surface

### Superuser flag

`User.is_superuser` is a boolean on the users table. Promote via SQL:

```sql
UPDATE users SET is_superuser = true WHERE email = 'ops@makstartup.com';
```

The flag is exposed on `GET /api/v1/auth/me` as `is_superuser` so the
frontend can show the "Admin" sidebar section. Backend admin endpoints
declare `Depends(require_superuser)` from `backend/app/core/auth.py`.

### Webhook replay

Stored as full raw payloads in `webhook_events.payload` (the unique
constraint `(provider, provider_event_id)` keeps idempotency). All four
provider webhooks now persist the entire payload so replay can re-run the
handler exactly.

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/v1/billing/admin/webhook-events?provider=&state=&limit=` | List recent events. `state ∈ {ok, error, pending}`. |
| `GET` | `/api/v1/billing/admin/webhook-events/{id}` | Full event including payload — for the JSON drawer. |
| `POST` | `/api/v1/billing/admin/webhook-events/{id}/replay` | Re-runs the provider-specific handler. Clears `error`, stamps `processed_at`. |

UI lives at **`/app/admin/webhooks`**:
- Provider + state filter pills
- Auto-refresh every 15 s
- Status dots (emerald ok / red error / pulsing amber pending)
- Right-side slide-in drawer with the full payload + "Replay event" button

Per-provider replay behaviour (`backend/app/services/webhook_replay_service.py`):

| Provider | Event types handled |
|---|---|
| Stripe | `customer.subscription.{created,updated,deleted}`, `checkout.session.completed` (re-fetches the live Stripe Subscription), `invoice.paid`, `invoice.payment_failed` |
| MTN | Successful transaction → `confirm_by_provider_id` |
| Airtel | Successful transaction → `confirm_by_provider_id` |
| Flutterwave | Successful transaction → `confirm_by_tx_ref` |

### Manual renewal-reminder trigger

```bash
curl -X POST https://www.optiscan.makstartup.com/api/v1/billing/admin/run-renewal-reminders \
     -H "Authorization: Bearer $ACCESS"
```

Same logic as the daily cron. Idempotent — re-running before
`current_period_end` advances is a no-op. Returns the
`RenewalRunResult` dict (`{found_subscriptions, reminders_sent, reminders_skipped, errors}`).

---

## 17. Free-tier self-serve downgrade

Triggered from **`/app/billing`** via the "Downgrade to Free" text-button
next to "Cancel at period end". Opens `DowngradeDialog`, which calls
`POST /api/v1/billing/subscription/change {plan_code: "free", billing_cycle: "monthly"}`.

Backend logic in `billing_service.change_plan_immediately`:

1. **Member-count guard** — if the org has more active members than the Free
   plan allows (1), returns:
   ```json
   {
     "error": "members_exceed_target_seats",
     "message": "You have 4 active members but the Free plan only supports 1. …",
     "active_members": 4,
     "target_plan": "free",
     "target_seat_limit": 1
   }
   ```
   The frontend renders `message` inline in the dialog so the user knows
   exactly how many seats to free.
2. **Stripe cancellation** — when the current subscription is Stripe-billed,
   calls `stripe.Subscription.delete(stripe_subscription_id, prorate=True)`
   so Stripe issues a prorated refund credit. Failure to cancel is logged
   but doesn't block the local flip — better to leave a stale Stripe sub
   for ops to clean up than to trap the user in a paid plan.
3. **Local cleanup** — zeroes `additional_seats`, clears
   `stripe_seat_item_id` and `stripe_subscription_id`, but keeps
   `stripe_customer_id` so a re-upgrade reuses the saved payment method.
4. **MoMo / Flutterwave** — current period is forfeit (those rails don't
   prorate refunds). The dialog calls this out explicitly.

---

## 18. Build log

Iterative additions in the order they shipped — useful when triaging
a regression or onboarding someone:

| # | Date | Surface |
|---|---|---|
| 1 | 2026-05-15 | **Phase A** — Postgres schema, SQLAlchemy models, Alembic migration 0001, plan seed |
| 2 | 2026-05-15 | **Phase B** — Auth (register / login / refresh / magic-link / reset / verify-email / me), Organization + Membership models, /api/v1/orgs CRUD |
| 3 | 2026-05-15 | **Phase C** — Quota dependency, feature-gate dependency, billing router (plans / subscription / usage / invoices) |
| 4 | 2026-05-15 | **Frontend route restructure** — `(public)` + `(app)` route groups, `proxy.ts`, sign-in / sign-up / verify-email / reset-password / contact-sales, /app/* file routes |
| 5 | 2026-05-15 | **Pricing + paywall UX** — `/pricing` with annual toggle + matrix, `PaywallModal`, `UpsellSheet`, `UsageChip`, `/app/usage`, `/app/billing` |
| 6 | 2026-05-15 | **Phase D Stripe** — checkout-session, billing portal, webhook with idempotency, `apply_stripe_subscription_event`, `apply_stripe_invoice_paid` |
| 7 | 2026-05-15 | **Feature gates rolled out** — router-level `Depends(require_tier(…))` on clinical / governance / dhis2 / fhir / dicom / sms / review / monitoring / agents |
| 8 | 2026-05-15 | **Phase E MoMo + Flutterwave** — momo_billing_service, signed callbacks with `webhook_events` idempotency, Flutterwave hosted checkout, intent polling on the frontend |
| 9 | 2026-05-15 | **Marketing UX upgrade** — animated hero with SVG fundus mockup, How It Works, value-props (6 cards), compliance badges, testimonials, FAQ with JSON-LD, scroll-aware nav, dark footer with newsletter |
| 10 | 2026-05-15 | **Domain swap** — canonical `https://www.optiscan.makstartup.com`, CORS for both www and non-www |
| 11 | 2026-05-15 | **Legal pages** — `/legal/privacy` (11 sections, PDP Act 2019), `/legal/terms` (14 sections, Uganda governing law), shared `LegalDoc` component |
| 12 | 2026-05-15 | **Email provider wiring** — `email_templates.py` (verification / magic-link / reset / invite / renewal-reminder), retry-aware Resend + SendGrid + SMTP backends |
| 13 | 2026-05-15 | **Renewal-reminder cron** — `renewal_reminders` table (migration 0002), `renewal_service.py`, CLI module, admin trigger endpoint |
| 14 | 2026-05-15 | **Renewal banner** — global `(app)/layout.tsx` slim strip + inline card on `/app/billing`, dismissible per `period_end` |
| 15 | 2026-05-15 | **Phase F seats** — `additional_seats` column (migration 0003), `seat_service.py`, `/api/v1/billing/seats` GET + POST, `SeatManager` UI on `/app/team`, Stripe seat-item lazy create + webhook quantity sync |
| 16 | 2026-05-15 | **Webhook replay ops view** — full raw payloads persisted, admin endpoints, `/app/admin/webhooks` page with table + drawer + replay button |
| 17 | 2026-05-15 | **Free-tier self-serve downgrade** — member-count guard, Stripe `prorate=True` cancellation, `DowngradeDialog` on `/app/billing` |
| 18 | 2026-05-15 | **Postgres Option 1 — embedded** — `postgres-bootstrap.sh`, `backend-start.sh`, in-image `postgresql-14`, supervisord priority-ordered programs |
| 19 | 2026-05-15 | **Postgres Option 2 — sidecar (current default)** — `postgres` service in docker-compose, k8s/base StatefulSet + headless Service + Secret, `EMBEDDED_POSTGRES__ENABLED` flag to silence the in-image cluster |
