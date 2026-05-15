"""Password hashing + JWT helpers.

Uses argon2 for password hashing (industry standard, OWASP-recommended) and
PyJWT for signed tokens. Refresh tokens are opaque random bytes — only the
SHA-256 hash is stored in `refresh_tokens` so a DB leak cannot replay sessions.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
import time
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

_hasher = PasswordHasher()


# ── Passwords ──

def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(plain: str, stored_hash: str) -> bool:
    try:
        return _hasher.verify(stored_hash, plain)
    except VerifyMismatchError:
        return False
    except Exception as exc:  # malformed hash
        logger.warning("Password verify error: %s", exc)
        return False


def password_needs_rehash(stored_hash: str) -> bool:
    return _hasher.check_needs_rehash(stored_hash)


# ── JWT (access tokens) ──

def encode_access_token(
    *,
    user_id: str,
    org_id: str | None,
    role: str = "user",
    ttl_seconds: int | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": user_id,
        "org": org_id,
        "role": role,
        "iat": now,
        "exp": now + (ttl_seconds or settings.jwt_access_ttl_seconds),
        "typ": "access",
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError as exc:
        logger.debug("Invalid JWT: %s", exc)
        return None


# ── Refresh tokens (opaque, hashed-at-rest) ──

def generate_refresh_token() -> str:
    """Return a fresh 256-bit URL-safe token. Caller must hash before DB write."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """Store this in DB; never store the raw token."""
    return hashlib.sha256(token.encode()).hexdigest()


# ── Single-use email tokens (magic link, verify email, password reset, invite) ──

def generate_email_token() -> str:
    return secrets.token_urlsafe(32)
