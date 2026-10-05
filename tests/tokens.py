"""Helpers that mint JWTs for tests. Test-only: the application never issues tokens."""

import time
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from app.core.config import Settings

ISSUER = "https://issuer.test"
AUDIENCE = "policy-rag"
SECRET = "test-secret-that-is-at-least-32-characters-long"


def settings(url: str = "postgresql://x/y", **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": url,
        "auth_mode": "jwt",
        "auth_issuer": ISSUER,
        "auth_audience": AUDIENCE,
        "auth_jwt_secret": SECRET,
        **overrides,
    }
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def claims(**overrides: Any) -> dict[str, Any]:
    now = int(time.time())
    base: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "user-1",
        "iat": now,
        "exp": now + 600,
        "roles": ["employee"],
        "tenant_id": "default",
        "department": [],
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def hs256(secret: str = SECRET, **overrides: Any) -> str:
    return jwt.encode(claims(**overrides), secret, algorithm="HS256")


def rsa_keypair() -> tuple[Any, str]:
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = (
        private.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    return private, public_pem


def rs256(private_key: Any, kid: str | None = None, **overrides: Any) -> str:
    headers = {"kid": kid} if kid else None
    return jwt.encode(claims(**overrides), private_key, algorithm="RS256", headers=headers)
