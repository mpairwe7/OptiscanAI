"""Authentication token endpoint."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.app.core.auth import create_access_token
from backend.app.core.config import settings

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


class TokenRequest(BaseModel):
    api_key: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/token", response_model=TokenResponse)
async def get_token(request: TokenRequest):
    """Exchange API key for JWT access token.
    In production, validate against a database of API keys."""
    if not settings.auth_enabled:
        return TokenResponse(
            access_token=create_access_token("anonymous", role="admin"),
            expires_in=settings.jwt_expiry_seconds,
        )

    # Simple API key validation (replace with DB lookup in production)
    if request.api_key != settings.jwt_secret:
        raise HTTPException(status_code=401, detail="Invalid API key")

    token = create_access_token(subject="api_user", role="user", expires_seconds=settings.jwt_expiry_seconds)
    return TokenResponse(access_token=token, expires_in=settings.jwt_expiry_seconds)
