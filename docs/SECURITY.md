# Security: authentication, authorization and data minimization

`AUTH_MODE=jwt` makes the API and the MCP server validate a bearer token and limit everything to what it allows. The default, `AUTH_MODE=off`, is the local demo mode: no login, and `/api/me` reports it.

In short: JWT validation (OIDC/JWKS, PEM key, or demo shared secret), roles, and document-level authorization enforced inside the retrieval SQL (tenant, access level, department); the HTTP API and the MCP server obey the same scope.

## Token validation (JWT, OIDC/JWKS)

Signature, issuer, audience, subject and expiry are required, with a small clock leeway. Missing or invalid tokens get `401`; a valid token with no recognised role gets `403`. Roles in a token that this project does not define are ignored. Misconfiguration (no key source, a short secret, missing issuer or audience) stops the server at startup rather than running it open.

Key sources (exactly one): `AUTH_JWKS_URL` (an OIDC provider's JWKS endpoint, the one to use in a real deployment), `AUTH_PUBLIC_KEY` (a PEM public key) or `AUTH_JWT_SECRET` (a shared secret, local demos only). This project does not issue tokens or run an identity provider.

### Algorithm allowlist

The algorithm allowlist comes from the configured key source, not from the token: a JWKS URL or PEM public key accepts only RS256/ES256, a shared secret (at least 32 characters, local demos only) only HS256, so `alg: none` and HS/RS key-confusion tokens are rejected.

## Roles and document labels

Every document (and each of its chunks) is labelled with a `tenant_id`, an `access_level` (`public` < `internal` < `restricted` < `confidential`) and a `department`, set in the document's front matter (unlabelled documents are `internal` in the `default` tenant).

A token's roles set the highest level its holder may read:

| Role | Highest readable level |
|---|---|
| `employee` | internal |
| `underwriter` | restricted |
| `compliance` | restricted |
| `admin` | confidential |

## Tenant, department and access-level rules

Public and internal documents are open to the whole tenant; restricted and confidential ones also require the holder's department to match (admins belong to every department). Other tenants' documents are never readable, and a chunk with missing labels is treated as unreadable.

## Retrieval-time enforcement

The scope is built only from the validated token and turned into a SQL predicate that is ANDed into the semantic, lexical and hybrid queries, the document listing and the document reader. Restricted chunks are therefore never fetched, so they cannot be ranked, reranked, scored by the evidence gate, handed to a generator, put in a response, or written to a trace or log.

Caller metadata filters are a separate, additional condition: they can narrow results but cannot widen the scope, `tenant_id` and the other authorization fields are not accepted as filters, and nothing in the request (query string, headers, body) can name a tenant.

## MCP security

`python -m app.mcp_server` with `AUTH_MODE=jwt` refuses to start without a valid `MCP_ACCESS_TOKEN` in its environment, and then every tool call is limited by that token's scope. It runs over stdio, so the client that launches the process supplies its own credential.

The tools take typed, length-limited arguments (a document name must match `[a-z0-9-]`, so paths and SQL are rejected before any code runs), have no SQL, filesystem, shell or network access, and are annotated read-only. Failures come back as MCP tool errors with a safe message; unexpected exceptions are masked; each call has a deadline (`MCP_TOOL_TIMEOUT_S`). The server can be started with fixed metadata filters that a caller can narrow but never change, and the same scope applies as for the HTTP API. Operating instructions are in [OPERATIONS.md](OPERATIONS.md#mcp-server).

## Data minimization

Telemetry is designed to investigate a bad answer without recording what users asked. What is never recorded: question text (only its length; a 120-character preview is available behind `RECORD_QUERY_TEXT=true`), document text, answers, request headers or bodies, and credentials. Attribute names that suggest secrets are dropped; credential-shaped strings are masked in logs; span errors carry the exception type, not its message. Tests assert these properties. See [OPERATIONS.md](OPERATIONS.md#observability) for what is recorded.

Generator providers follow the same rule: Bedrock credentials come from the normal AWS chain and are never read from this project's settings or committed; provider errors are translated to a safe 502 that carries no credentials or request details.

## Local demo tokens

`python -m scripts.make_demo_token --role underwriter --department underwriting` mints a short-lived HS256 token from `AUTH_JWT_SECRET`, for demos and tests only.

```bash
export AUTH_MODE=jwt AUTH_ISSUER=demo AUTH_AUDIENCE=policy-rag AUTH_JWT_SECRET="$(openssl rand -hex 32)"
docker compose up -d --build && docker compose exec -T api python -m scripts.ingest
TOKEN=$(docker compose exec -T api python -m scripts.make_demo_token --role employee 2>/dev/null)
curl -s localhost:8000/api/me -H "Authorization: Bearer $TOKEN"
```

## Cloud deployment

The [AWS reference architecture](../infra/terraform/README.md) makes JWT authentication mandatory (no demo mode), verifies tokens against the identity provider's public keys, and keeps the database password in Secrets Manager.
