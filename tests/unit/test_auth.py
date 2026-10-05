"""JWT validation: what is accepted, and the many ways a token can be wrong."""

import base64
import hashlib
import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import jwt
import pytest
from jwt.algorithms import RSAAlgorithm

from app.security.auth import AuthError, TokenValidator
from tests import tokens
from tests.tokens import claims, hs256, rs256, rsa_keypair


def forge_hs256_with(key: bytes, payload: dict[str, Any]) -> str:
    """Build an HS256 token by hand, signing with `key` (PyJWT refuses to do this for a PEM)."""

    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    signing_input = b64(b'{"alg":"HS256","typ":"JWT"}') + "." + b64(json.dumps(payload).encode())
    signature = hmac.new(key, signing_input.encode(), hashlib.sha256).digest()
    return signing_input + "." + b64(signature)


def validator(**overrides: Any) -> TokenValidator:
    return TokenValidator(tokens.settings(**overrides))


def expect(code: str, token: str, v: TokenValidator | None = None) -> None:
    with pytest.raises(AuthError) as raised:
        (v or validator()).validate(token)
    assert raised.value.code == code
    if token:
        assert token not in str(raised.value)
        assert token not in raised.value.detail


# --- accepted tokens ---------------------------------------------------------------------------


def test_a_valid_token_yields_the_principal() -> None:
    principal = validator().validate(
        hs256(roles=["Underwriter"], department=["Underwriting"], tenant_id="acme", sub="u-9")
    )
    assert principal.subject == "u-9"
    assert principal.roles == ("underwriter",)
    assert principal.departments == ("underwriting",)
    assert principal.tenant_id == "acme"


def test_roles_may_be_a_space_or_comma_separated_string() -> None:
    assert validator().validate(hs256(roles="employee underwriter")).roles == (
        "employee",
        "underwriter",
    )
    assert validator().validate(hs256(roles="employee,compliance")).roles == (
        "employee",
        "compliance",
    )


def test_claim_names_are_configurable() -> None:
    v = validator(auth_roles_claim="groups", auth_tenant_claim="org", auth_department_claim="dept")
    token = jwt.encode(
        claims(groups=["admin"], org="acme", dept="legal", roles=None, tenant_id=None),
        tokens.SECRET,
        algorithm="HS256",
    )
    principal = v.validate(token)
    assert (principal.roles, principal.tenant_id, principal.departments) == (
        ("admin",),
        "acme",
        ("legal",),
    )


def test_a_token_that_grants_no_known_role_authenticates_but_has_no_scope() -> None:
    v = validator()
    assert v.validate(hs256(roles=["janitor"])).roles == ("janitor",)
    with pytest.raises(AuthError) as raised:
        v.scope(hs256(roles=["janitor"]))
    assert raised.value.code == "no_role"


# --- rejected tokens ---------------------------------------------------------------------------


def test_an_expired_token_is_rejected() -> None:
    expect("expired", hs256(exp=int(time.time()) - 3600))


def test_a_small_clock_skew_is_tolerated_but_not_a_large_one() -> None:
    validator().validate(hs256(exp=int(time.time()) - 5))  # within the 30 s leeway
    expect("expired", hs256(exp=int(time.time()) - 120))


def test_a_token_that_is_not_yet_valid_is_rejected() -> None:
    expect("invalid_token", hs256(nbf=int(time.time()) + 3600))


@pytest.mark.parametrize("claim", ["exp", "iss", "aud", "sub"])
def test_missing_required_claims_are_rejected(claim: str) -> None:
    token = jwt.encode(
        {k: v for k, v in claims().items() if k != claim}, tokens.SECRET, algorithm="HS256"
    )
    expect("invalid_token", token)


def test_the_wrong_issuer_or_audience_is_rejected() -> None:
    expect("invalid_token", hs256(iss="https://evil.example"))
    expect("invalid_token", hs256(aud="another-service"))


def test_a_wrong_signature_is_rejected() -> None:
    expect("invalid_token", hs256(secret="a-different-secret-that-is-also-32-chars!!"))


def test_a_tampered_payload_is_rejected() -> None:
    header, payload, signature = hs256(roles=["employee"]).split(".")
    forged = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    forged["roles"] = ["admin"]
    new_payload = base64.urlsafe_b64encode(json.dumps(forged).encode()).rstrip(b"=").decode()
    expect("invalid_token", f"{header}.{new_payload}.{signature}")


def test_unsigned_alg_none_tokens_are_rejected() -> None:
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').rstrip(b"=").decode()
    body = (
        base64.urlsafe_b64encode(json.dumps(claims(roles=["admin"])).encode()).rstrip(b"=").decode()
    )
    expect("invalid_token", f"{header}.{body}.")


@pytest.mark.parametrize("garbage", ["", "not-a-jwt", "a.b.c", "....", "x" * 500])
def test_malformed_tokens_are_rejected(garbage: str) -> None:
    expect("invalid_token", garbage)


def test_a_hmac_token_signed_with_the_public_key_is_rejected_by_an_rsa_validator() -> None:
    """The classic algorithm-confusion attack: sign HS256 with the (public) RSA key as secret."""
    _, public_pem = rsa_keypair()
    v = validator(auth_jwt_secret=None, auth_public_key=public_pem)
    forged = forge_hs256_with(public_pem.encode(), claims(roles=["admin"]))
    expect("invalid_token", forged, v)


def test_an_rsa_token_is_rejected_by_a_shared_secret_validator() -> None:
    private, _ = rsa_keypair()
    expect("invalid_token", rs256(private))


def test_an_rsa_validator_accepts_tokens_from_the_matching_private_key_only() -> None:
    private, public_pem = rsa_keypair()
    other, _ = rsa_keypair()
    v = validator(auth_jwt_secret=None, auth_public_key=public_pem)
    assert v.validate(rs256(private, roles=["admin"])).roles == ("admin",)
    expect("invalid_token", rs256(other), v)


@pytest.mark.parametrize(
    "tenant", ["", "has space", "a/b", "x" * 65, "t'; DROP TABLE x; --", 5, None]
)
def test_the_tenant_claim_must_be_a_plain_identifier(tenant: Any) -> None:
    token = jwt.encode(claims(tenant_id=tenant), tokens.SECRET, algorithm="HS256")
    expect("invalid_claims", token)


# --- configuration is checked up front ---------------------------------------------------------


def test_a_short_shared_secret_is_refused() -> None:
    with pytest.raises(ValueError, match="32 characters"):
        validator(auth_jwt_secret="too-short")


def test_exactly_one_key_source_is_required() -> None:
    with pytest.raises(ValueError, match="exactly one key source"):
        validator(auth_jwt_secret=None)
    _, public_pem = rsa_keypair()
    with pytest.raises(ValueError, match="exactly one key source"):
        validator(auth_public_key=public_pem)  # a secret is configured too


def test_issuer_and_audience_are_required() -> None:
    with pytest.raises(ValueError, match="AUTH_ISSUER"):
        validator(auth_issuer=None)
    with pytest.raises(ValueError, match="AUTH_AUDIENCE"):
        validator(auth_audience=None)


# --- JWKS (OIDC-style key discovery), against a local key server -------------------------------


class JwksServer:
    def __init__(self, keys: list[dict[str, Any]]) -> None:
        body = json.dumps({"keys": keys}).encode()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/jwks.json"

    def close(self) -> None:
        self.server.shutdown()


def jwk(private: Any, kid: str) -> dict[str, Any]:
    key = json.loads(RSAAlgorithm.to_jwk(private.public_key()))
    key.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return key


def test_signing_keys_are_discovered_from_a_jwks_endpoint_by_kid() -> None:
    private, _ = rsa_keypair()
    server = JwksServer([jwk(private, "key-1")])
    try:
        v = validator(auth_jwt_secret=None, auth_jwks_url=server.url)
        assert v.validate(rs256(private, kid="key-1", roles=["compliance"])).roles == (
            "compliance",
        )
        expect("unknown_key", rs256(private, kid="rotated-away"), v)
        other, _ = rsa_keypair()
        expect("invalid_token", rs256(other, kid="key-1"), v)  # right kid, wrong key
    finally:
        server.close()
