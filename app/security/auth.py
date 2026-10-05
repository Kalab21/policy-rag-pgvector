"""JWT validation. The project does not issue tokens or run an identity provider: it validates
tokens that an OIDC-style provider (or, for local demos, `scripts/make_demo_token.py`) issued.

Validation is strict. The signature algorithm must be on an allowlist chosen from the key
source (never from the token), so `alg: none` and HS/RS key-confusion tokens are rejected;
issuer, audience, subject and expiry are required; and the claims that drive authorization
(roles, tenant, department) are read, normalised and bounded here, in one place.
"""

import re
from dataclasses import dataclass
from typing import Any

import jwt
from jwt import PyJWKClient

from app.core.config import Settings
from app.security.access import AccessScope, scope_for

_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_MAX_LIST = 20


class AuthError(Exception):
    """Authentication failed. `code` is a short machine-readable reason for logs and metrics;
    `detail` is safe to show the caller. Neither ever contains the token."""

    def __init__(self, code: str, detail: str = "invalid or missing credentials") -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class Principal:
    subject: str
    roles: tuple[str, ...]
    tenant_id: str
    departments: tuple[str, ...]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return value.replace(",", " ").split()
    if isinstance(value, list):
        return [str(v) for v in value[:_MAX_LIST] if isinstance(v, str)]
    return []


class TokenValidator:
    """Validates bearer tokens against one configured key source."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._jwks: PyJWKClient | None = None
        sources = [
            bool(settings.auth_jwks_url),
            bool(settings.auth_public_key),
            settings.auth_jwt_secret is not None,
        ]
        if sum(sources) != 1:
            raise ValueError(
                "AUTH_MODE=jwt needs exactly one key source: AUTH_JWKS_URL, "
                "AUTH_PUBLIC_KEY or AUTH_JWT_SECRET"
            )
        if not settings.auth_issuer or not settings.auth_audience:
            raise ValueError("AUTH_MODE=jwt needs AUTH_ISSUER and AUTH_AUDIENCE")
        if settings.auth_jwt_secret is not None:
            if len(settings.auth_jwt_secret.get_secret_value()) < 32:
                raise ValueError("AUTH_JWT_SECRET must be at least 32 characters")
            self._algorithms = ["HS256"]  # a shared secret: symmetric only
        else:
            self._algorithms = ["RS256", "ES256"]  # a public key: asymmetric only
        if settings.auth_jwks_url:
            self._jwks = PyJWKClient(settings.auth_jwks_url, cache_keys=True, timeout=5)

    def _key(self, token: str) -> Any:
        settings = self._settings
        if settings.auth_jwt_secret is not None:
            return settings.auth_jwt_secret.get_secret_value()
        if settings.auth_public_key:
            return settings.auth_public_key
        assert self._jwks is not None
        try:
            return self._jwks.get_signing_key_from_jwt(token).key
        except jwt.PyJWKClientError:
            raise AuthError("unknown_key") from None

    def validate(self, token: str) -> Principal:
        settings = self._settings
        try:
            claims = jwt.decode(
                token,
                self._key(token),
                algorithms=self._algorithms,
                audience=settings.auth_audience,
                issuer=settings.auth_issuer,
                leeway=settings.auth_leeway_s,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.ExpiredSignatureError:
            raise AuthError("expired") from None
        except AuthError:
            raise
        except jwt.PyJWTError:
            # Bad signature, wrong issuer/audience, malformed, wrong algorithm, missing claim.
            raise AuthError("invalid_token") from None

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject or len(subject) > 200:
            raise AuthError("invalid_claims")
        tenant = claims.get(settings.auth_tenant_claim)
        if not isinstance(tenant, str) or not _IDENTIFIER.match(tenant):
            raise AuthError("invalid_claims", "the token has no valid tenant")
        roles = tuple(r.lower() for r in _as_list(claims.get(settings.auth_roles_claim)))
        departments = tuple(d.lower() for d in _as_list(claims.get(settings.auth_department_claim)))
        return Principal(subject, roles, tenant, departments)

    def scope(self, token: str) -> AccessScope:
        principal = self.validate(token)
        scope = scope_for(principal.roles, principal.tenant_id, principal.departments)
        if scope is None:
            raise AuthError("no_role", "the token grants no role that can read policies")
        return scope
