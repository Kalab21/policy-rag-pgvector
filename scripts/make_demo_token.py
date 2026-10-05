"""Mint a JWT for LOCAL DEMOS and tests. This is not an identity provider.

    AUTH_MODE=jwt AUTH_ISSUER=demo AUTH_AUDIENCE=policy-rag AUTH_JWT_SECRET=<32+ chars> \\
        python -m scripts.make_demo_token --role underwriter --department underwriting

It signs a short-lived HS256 token with the shared secret from the environment, using the issuer,
audience and claim names the server is configured with, and prints only the token on stdout.
A real deployment validates tokens from an OIDC provider (AUTH_JWKS_URL) and never needs this.
"""

import argparse
import sys
import time

import jwt

from app.core.config import get_settings
from app.security.access import ROLES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--role", action="append", required=True, choices=ROLES)
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--department", action="append", default=[])
    parser.add_argument("--sub", default="demo-user")
    parser.add_argument("--ttl-minutes", type=int, default=60)
    args = parser.parse_args()

    settings = get_settings()
    if settings.auth_jwt_secret is None or not settings.auth_issuer or not settings.auth_audience:
        print(
            "set AUTH_JWT_SECRET, AUTH_ISSUER and AUTH_AUDIENCE (the same values the server uses)",
            file=sys.stderr,
        )
        return 2
    if not 1 <= args.ttl_minutes <= 1440:
        print("--ttl-minutes must be between 1 and 1440", file=sys.stderr)
        return 2
    now = int(time.time())
    claims = {
        "iss": settings.auth_issuer,
        "aud": settings.auth_audience,
        "sub": args.sub,
        "iat": now,
        "exp": now + args.ttl_minutes * 60,
        settings.auth_roles_claim: args.role,
        settings.auth_tenant_claim: args.tenant,
        settings.auth_department_claim: args.department,
    }
    print("demo token: local use only, not a production credential", file=sys.stderr)
    print(jwt.encode(claims, settings.auth_jwt_secret.get_secret_value(), algorithm="HS256"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
