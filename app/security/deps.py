"""FastAPI dependency that turns a bearer token into the caller's access scope."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from app.observability.logs import log_event
from app.observability.telemetry import current
from app.security.access import AccessScope
from app.security.auth import AuthError

MAX_TOKEN_CHARS = 8192


def _reject(code: str, status: int, detail: str) -> HTTPException:
    current().record("auth_failures", 1, {"reason": code})
    log_event("auth.failed", level=30, reason=code, status=status)  # never the token
    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
    return HTTPException(status_code=status, detail=detail, headers=headers)


def access_scope(request: Request) -> AccessScope | None:
    """None in demo mode (AUTH_MODE=off); otherwise the validated caller's scope.

    401 for a missing or invalid token, 403 for a valid token that grants no usable role.
    """
    if request.app.state.settings.auth_mode == "off":
        return None
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token or len(token) > MAX_TOKEN_CHARS:
        raise _reject("missing_token", 401, "invalid or missing credentials")
    try:
        return request.app.state.validator.scope(token)
    except AuthError as exc:
        raise _reject(exc.code, 403 if exc.code == "no_role" else 401, exc.detail) from None


CallerScope = Annotated[AccessScope | None, Depends(access_scope)]
