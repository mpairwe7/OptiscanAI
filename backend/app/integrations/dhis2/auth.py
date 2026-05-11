"""DHIS2 authentication handler.

Supports OAuth2 (preferred) and Personal Access Token (fallback).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class DHIS2Auth:
    """DHIS2 authentication handler."""

    def __init__(
        self,
        method: str = "pat",
        personal_access_token: str = "",
        oauth2_client_id: str = "",
        oauth2_client_secret: str = "",
        base_url: str = "",
    ):
        self._method = method
        self._pat = personal_access_token
        self._client_id = oauth2_client_id
        self._client_secret = oauth2_client_secret
        self._base_url = base_url
        self._oauth_token: Optional[str] = None
        self._token_expires: float = 0

    async def get_headers(self) -> dict[str, str]:
        """Get authorization headers."""
        if self._method == "oauth2":
            token = await self._get_oauth_token()
            return {"Authorization": f"Bearer {token}"}
        return {"Authorization": f"ApiToken {self._pat}"}

    async def _get_oauth_token(self) -> str:
        """Get or refresh OAuth2 token."""
        if self._oauth_token and time.time() < self._token_expires:
            return self._oauth_token

        url = f"{self._base_url}/uaa/oauth/token"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
            ) as resp:
                data = await resp.json()
                self._oauth_token = data["access_token"]
                self._token_expires = time.time() + data.get("expires_in", 3600) - 60
                return self._oauth_token
