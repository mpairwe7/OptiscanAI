"""Outbound email — pluggable provider behind a single async send_email.

Providers:
  - ``console`` (default) — log the email; never sends. Used in dev/tests.
  - ``smtp`` — SMTP relay via ``aiosmtplib``.
  - ``resend`` — Resend HTTP API.
  - ``sendgrid`` — SendGrid Mail Send API.

All HTTP providers retry once on a 5xx response or transient network error to
keep transactional flows resilient. Failures are logged but do not raise to the
caller — auth/registration flows shouldn't 500 because email is down.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from backend.app.core.config import settings
from backend.app.services.email_templates import RenderedEmail

logger = logging.getLogger(__name__)


# Backwards-compat: app_url is imported from this module by older callers.
def app_url(path: str) -> str:
    base = settings.public_app_url.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"


async def send_email(
    *,
    to: str,
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
) -> None:
    """Send an email through the configured provider.

    Never raises — provider errors are logged so a transient SMTP/Resend outage
    doesn't break account registration.
    """
    provider = (settings.email.provider or "console").lower()
    if not settings.email.enabled or provider == "console":
        logger.info(
            "[email:console] to=%s subject=%s\n%s",
            to,
            subject,
            body_text,
        )
        return

    try:
        if provider == "smtp":
            await _send_smtp(to, subject, body_text, body_html)
        elif provider == "resend":
            await _retry(_send_resend, to, subject, body_text, body_html)
        elif provider == "sendgrid":
            await _retry(_send_sendgrid, to, subject, body_text, body_html)
        else:
            logger.warning(
                "Unknown email provider %r — falling back to console log",
                provider,
            )
            logger.info("[email:fallback] to=%s subject=%s\n%s", to, subject, body_text)
    except Exception as exc:
        logger.error("Email send failed (to=%s, subject=%s): %s", to, subject, exc)


async def send_rendered(*, to: str, email: RenderedEmail) -> None:
    await send_email(
        to=to,
        subject=email.subject,
        body_text=email.body_text,
        body_html=email.body_html,
    )


# ── Retry helper ──

async def _retry(fn, *args, attempts: int = 2, backoff_s: float = 1.5) -> None:
    last_exc: Optional[BaseException] = None
    for i in range(attempts):
        try:
            await fn(*args)
            return
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            last_exc = exc
            if i < attempts - 1:
                await asyncio.sleep(backoff_s * (i + 1))
    if last_exc is not None:
        raise last_exc


# ── Provider implementations ──

def _from_header() -> str:
    return f"{settings.email.from_name} <{settings.email.from_address}>"


async def _send_smtp(to: str, subject: str, body_text: str, body_html: Optional[str]) -> None:
    from email.message import EmailMessage

    import aiosmtplib  # type: ignore  # optional dep

    msg = EmailMessage()
    msg["From"] = _from_header()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body_text)
    if body_html:
        msg.add_alternative(body_html, subtype="html")

    await aiosmtplib.send(
        msg,
        hostname=settings.email.smtp_host,
        port=settings.email.smtp_port,
        username=settings.email.smtp_username or None,
        password=settings.email.smtp_password or None,
        start_tls=True,
    )


async def _send_resend(to: str, subject: str, body_text: str, body_html: Optional[str]) -> None:
    if not settings.email.resend_api_key:
        raise RuntimeError("Resend not configured — set EMAIL__RESEND_API_KEY")
    payload: dict[str, object] = {
        "from": _from_header(),
        "to": [to],
        "subject": subject,
        "text": body_text,
    }
    if body_html:
        payload["html"] = body_html
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {settings.email.resend_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        # Retry only on 5xx — 4xx is our fault (bad address) and shouldn't loop
        if r.status_code >= 500:
            r.raise_for_status()
        if r.status_code >= 400:
            logger.warning("Resend 4xx: %s — %s", r.status_code, r.text)
            return


async def _send_sendgrid(to: str, subject: str, body_text: str, body_html: Optional[str]) -> None:
    if not settings.email.sendgrid_api_key:
        raise RuntimeError("SendGrid not configured — set EMAIL__SENDGRID_API_KEY")
    content: list[dict[str, str]] = [{"type": "text/plain", "value": body_text}]
    if body_html:
        content.append({"type": "text/html", "value": body_html})
    payload = {
        "personalizations": [{"to": [{"email": to}]}],
        "from": {"email": settings.email.from_address, "name": settings.email.from_name},
        "subject": subject,
        "content": content,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {settings.email.sendgrid_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if r.status_code >= 500:
            r.raise_for_status()
        if r.status_code >= 400:
            logger.warning("SendGrid 4xx: %s — %s", r.status_code, r.text)
            return
