"""JWT authentication for API endpoints."""
import time
import hashlib
import hmac
import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


class TokenPayload(BaseModel):
    sub: str
    exp: float
    role: str = "user"


def _encode_token(payload: dict, secret: str) -> str:
    """Simple HMAC-based token (production: use PyJWT or python-jose)."""
    import json, base64
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    signature = hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).hexdigest()
    return f"{header}.{body}.{signature}"


def _decode_token(token: str, secret: str) -> Optional[dict]:
    """Decode and verify HMAC token."""
    import json, base64
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header, body, signature = parts
        expected = hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        # Pad base64
        body_padded = body + "=" * (4 - len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(body_padded))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except (ValueError, KeyError, json.JSONDecodeError, Exception) as e:
        logger.debug("Token decode failed: %s", e)
        return None


def create_access_token(subject: str, role: str = "user", expires_seconds: int = 3600) -> str:
    """Create a signed access token."""
    payload = {"sub": subject, "role": role, "exp": time.time() + expires_seconds, "iat": time.time()}
    return _encode_token(payload, settings.jwt_secret)


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[TokenPayload]:
    """Extract and validate JWT from Authorization header.
    Returns None if auth is disabled (settings.auth_enabled=False)."""
    if not settings.auth_enabled:
        return TokenPayload(sub="anonymous", exp=time.time() + 3600, role="admin")

    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authorization header")

    payload = _decode_token(credentials.credentials, settings.jwt_secret)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    return TokenPayload(**{k: payload[k] for k in ["sub", "exp", "role"] if k in payload})


def require_role(required_role: str):
    """Dependency that enforces a specific role."""
    async def role_checker(user: TokenPayload = Depends(get_current_user)):
        if user.role != required_role and user.role != "admin":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Role '{required_role}' required")
        return user
    return role_checker
