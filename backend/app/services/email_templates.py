"""Transactional email templates — branded HTML + plain-text variants.

Templates are pure-Python functions so we keep the dependency footprint flat
(no Jinja) and so tests can assert on returned bodies.

All templates return a :class:`RenderedEmail` carrying:
  - subject
  - body_text (always required, used as fallback for clients that strip HTML)
  - body_html (None for plain-only)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.app.core.config import settings


@dataclass
class RenderedEmail:
    subject: str
    body_text: str
    body_html: Optional[str]


# ── Layout helpers ──

_BRAND_COLOR = "#0d9488"  # accent teal
_BRAND_DARK = "#0f172a"


def _html_layout(*, heading: str, intro: str, cta_label: Optional[str], cta_url: Optional[str], body_html: str, support_email: str = "support@makstartup.com") -> str:
    cta_block = (
        f"""
        <table role="presentation" cellpadding="0" cellspacing="0" border="0" style="margin: 24px 0;">
          <tr>
            <td>
              <a href="{cta_url}" style="display:inline-block;padding:12px 24px;background:{_BRAND_COLOR};color:#ffffff;text-decoration:none;border-radius:8px;font-weight:600;font-size:14px;font-family:system-ui,-apple-system,Segoe UI,sans-serif;">
                {cta_label}
              </a>
            </td>
          </tr>
        </table>
        """
        if cta_label and cta_url
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width" />
    <title>{heading}</title>
  </head>
  <body style="margin:0;padding:0;background:#f8fafc;font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#0f172a;-webkit-font-smoothing:antialiased;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
      <tr>
        <td align="center" style="padding:32px 16px;">
          <table role="presentation" width="560" cellpadding="0" cellspacing="0" border="0" style="max-width:560px;width:100%;background:#ffffff;border-radius:16px;border:1px solid #e2e8f0;overflow:hidden;">
            <!-- Brand bar -->
            <tr>
              <td style="background:linear-gradient(90deg,{_BRAND_COLOR},#06b6d4);padding:18px 28px;color:#ffffff;">
                <div style="font-weight:700;letter-spacing:.01em;font-size:16px;">OptiscanAI</div>
                <div style="font-size:11px;opacity:.85;margin-top:2px;">Clinical retinal screening, explained.</div>
              </td>
            </tr>
            <!-- Body -->
            <tr>
              <td style="padding:32px 28px 8px;">
                <h1 style="margin:0 0 12px;font-size:22px;line-height:1.25;color:{_BRAND_DARK};font-weight:700;">{heading}</h1>
                <p style="margin:0 0 18px;font-size:15px;line-height:1.55;color:#334155;">{intro}</p>
                {body_html}
                {cta_block}
              </td>
            </tr>
            <!-- Footer -->
            <tr>
              <td style="padding:16px 28px 28px;border-top:1px solid #f1f5f9;background:#f8fafc;">
                <p style="margin:0;font-size:12px;color:#64748b;line-height:1.5;">
                  Questions? Reply to this email or write to
                  <a href="mailto:{support_email}" style="color:{_BRAND_COLOR};text-decoration:none;">{support_email}</a>.
                </p>
                <p style="margin:8px 0 0;font-size:11px;color:#94a3b8;">
                  OptiscanAI · A MakStartup project · Plot 51, Makerere Hill Road, Kampala, Uganda
                </p>
              </td>
            </tr>
          </table>
          <p style="margin:16px 0 0;font-size:11px;color:#94a3b8;font-family:system-ui,-apple-system,Segoe UI,sans-serif;">
            You received this because there is activity on your OptiscanAI account.
          </p>
        </td>
      </tr>
    </table>
  </body>
</html>
"""


_HONORIFICS = {"dr", "dr.", "prof", "prof.", "professor", "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "mx", "mx."}


def _first_name(full_name: str) -> str:
    """Best-effort first-name extraction that skips a leading honorific."""
    parts = [p for p in full_name.strip().split() if p]
    if not parts:
        return ""
    if parts[0].lower() in _HONORIFICS and len(parts) > 1:
        return parts[1]
    return parts[0]


def _app_url(path: str) -> str:
    base = settings.public_app_url.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"


# ── Templates ──


def email_verification(*, full_name: Optional[str], token: str) -> RenderedEmail:
    greeting = f"Hi {_first_name(full_name)}," if full_name else "Welcome,"
    link = _app_url(f"/verify-email?token={token}")
    text = (
        f"{greeting}\n\n"
        "Thanks for creating an OptiscanAI account.\n\n"
        f"Verify your email to activate it:\n{link}\n\n"
        "This link expires in 24 hours.\n\n"
        "If you didn't sign up, you can safely ignore this email."
    )
    html = _html_layout(
        heading="Verify your email",
        intro=f"{greeting}<br/>Thanks for creating an OptiscanAI account. Click below to verify your email and activate clinical screening.",
        cta_label="Verify email",
        cta_url=link,
        body_html='<p style="margin:0;font-size:13px;color:#64748b;">This link expires in 24 hours. If you didn&rsquo;t sign up, you can safely ignore this email.</p>',
    )
    return RenderedEmail(subject="Verify your OptiscanAI account", body_text=text, body_html=html)


def magic_link(*, email: str, token: str) -> RenderedEmail:
    link = _app_url(f"/sign-in/callback?token={token}")
    text = (
        "You requested a sign-in link for OptiscanAI.\n\n"
        f"Sign in:\n{link}\n\n"
        "This link expires in 15 minutes and can only be used once.\n\n"
        f"If you didn't request this, you can safely ignore it.\nEmail: {email}"
    )
    html = _html_layout(
        heading="Sign in to OptiscanAI",
        intro=f"Click below to sign in to your OptiscanAI account (<b>{email}</b>).",
        cta_label="Sign in",
        cta_url=link,
        body_html='<p style="margin:0;font-size:13px;color:#64748b;">This link expires in 15 minutes and can only be used once. If you didn&rsquo;t request it, ignore this email.</p>',
    )
    return RenderedEmail(subject="Sign in to OptiscanAI", body_text=text, body_html=html)


def password_reset(*, full_name: Optional[str], token: str) -> RenderedEmail:
    greeting = f"Hi {_first_name(full_name)}," if full_name else "Hi,"
    link = _app_url(f"/reset-password?token={token}")
    text = (
        f"{greeting}\n\n"
        "We received a request to reset your OptiscanAI password.\n\n"
        f"Reset it here:\n{link}\n\n"
        "This link expires in 1 hour. If you didn't request a reset, ignore this email — "
        "your password won't change."
    )
    html = _html_layout(
        heading="Reset your password",
        intro=f"{greeting}<br/>We received a request to reset your OptiscanAI password.",
        cta_label="Reset password",
        cta_url=link,
        body_html='<p style="margin:0;font-size:13px;color:#64748b;">This link expires in 1 hour. If you didn&rsquo;t request a reset, you can ignore this email &mdash; your password won&rsquo;t change.</p>',
    )
    return RenderedEmail(subject="Reset your OptiscanAI password", body_text=text, body_html=html)


def org_invite(*, inviter_name: Optional[str], inviter_email: str, organization_name: str, role: str, token: str) -> RenderedEmail:
    who = inviter_name or inviter_email
    link = _app_url(f"/sign-up?invite={token}")
    text = (
        f"{who} invited you to join {organization_name} on OptiscanAI as a {role}.\n\n"
        f"Accept the invite:\n{link}\n\n"
        "This invite expires in 7 days."
    )
    html = _html_layout(
        heading=f"Join {organization_name} on OptiscanAI",
        intro=f"<b>{who}</b> invited you to join <b>{organization_name}</b> as a <b>{role}</b>.",
        cta_label="Accept invite",
        cta_url=link,
        body_html='<p style="margin:0;font-size:13px;color:#64748b;">This invite expires in 7 days. If you weren&rsquo;t expecting this, you can safely ignore it.</p>',
    )
    return RenderedEmail(subject=f"You're invited to {organization_name} on OptiscanAI", body_text=text, body_html=html)


def renewal_reminder(
    *,
    full_name: Optional[str],
    organization_name: str,
    plan_display_name: str,
    period_end_iso: str,
    days_remaining: int,
    plan_code: str,
    billing_cycle: str,
    amount_usd: float,
) -> RenderedEmail:
    greeting = f"Hi {_first_name(full_name)}," if full_name else "Hi,"
    period_end = period_end_iso.split("T")[0]
    link = _app_url(f"/app/checkout/{plan_code}?cycle={billing_cycle}")
    if days_remaining <= 0:
        urgency = "expired today"
        subject = f"Your {plan_display_name} plan needs renewal"
    elif days_remaining == 1:
        urgency = "expires tomorrow"
        subject = f"Your {plan_display_name} plan expires tomorrow"
    else:
        urgency = f"expires in {days_remaining} days"
        subject = f"Renew your {plan_display_name} plan ({days_remaining} days left)"

    text = (
        f"{greeting}\n\n"
        f"Your {plan_display_name} subscription for {organization_name} {urgency} ({period_end}).\n\n"
        "Mobile-money subscriptions don't auto-renew — to keep clinical screening uninterrupted, "
        "renew with one click:\n"
        f"{link}\n\n"
        f"Amount: ${amount_usd:.2f} for one {billing_cycle} cycle.\n\n"
        "If you've already renewed via a different method, ignore this email."
    )
    html = _html_layout(
        heading=f"Renew {plan_display_name} for {organization_name}",
        intro=(
            f"{greeting}<br/>Your <b>{plan_display_name}</b> subscription <b>{urgency}</b> ({period_end}). "
            "Mobile-money subscriptions don&rsquo;t auto-renew, so one click here keeps clinical screening uninterrupted."
        ),
        cta_label=f"Renew — ${amount_usd:.2f}",
        cta_url=link,
        body_html=f'<p style="margin:0;font-size:13px;color:#64748b;">One {billing_cycle} cycle &middot; ${amount_usd:.2f} USD (charged in UGX at the displayed FX rate). If you&rsquo;ve already renewed, you can ignore this email.</p>',
    )
    return RenderedEmail(subject=subject, body_text=text, body_html=html)
