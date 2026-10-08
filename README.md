# Policy RAG Platform

**Secure Retrieval & Grounded AI Platform**

A retrieval-augmented generation (RAG) service that answers questions about **synthetic** lending-policy documents only from the documents the caller is authorized to read, cites its sources, and refuses when the evidence is insufficient. Built with FastAPI, PostgreSQL 16 + pgvector (HNSW), hybrid retrieval with Reciprocal Rank Fusion, cross-encoder reranking, LangGraph, MCP, retrieval-time authorization, a held-out evaluation and OpenTelemetry. It runs end to end with no paid API: a local embedding model and an extractive answer generator by default.

## What it demonstrates

- **Retrieval:** semantic search (pgvector HNSW), PostgreSQL full-text search and hybrid search with Reciprocal Rank Fusion, plus optional cross-encoder reranking
- **Security:** retrieval-time JWT/RBAC document authorization: with authentication on, restricted documents are never retrieved, ranked or sent to a generator
- **Grounding:** a LangGraph flow with an evidence gate, citation validation and refusal of unsupported questions
- **Interfaces and operations:** a REST API, an MCP tool server, OpenTelemetry traces, metrics and structured logs
- **Evaluation:** separate tuning and held-out question sets (Hit@K, Recall@K, MRR, nDCG) with regression floors in CI
- **Engineering:** 490 automated tests, 217 of them against real PostgreSQL + pgvector

## Architecture

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/architecture-dark.svg">
    <img src="docs/architecture.svg" alt="Policy RAG Platform logical architecture. A REST API client or MCP consumer sends a bearer token. JWT validation (OIDC/JWKS or PEM key, algorithm allowlist) produces an access scope of roles, tenant, department and highest access level. Inside the authorized scope, the retrieval layer runs semantic (pgvector HNSW), lexical (PostgreSQL full-text) or hybrid (Reciprocal Rank Fusion) search against PostgreSQL 16 with pgvector, with the scope applied inside every SQL query; optional cross-encoder reranking follows. An evidence gate refuses when similarity is insufficient. Inside the AI generation boundary, LangGraph generation (extractive by default, Bedrock or OpenAI-compatible optional) and citation validation return an answer with sources or a refusal. OpenTelemetry, Prometheus metrics with JSON logs, held-out evaluation and GitHub Actions CI run across the platform." width="1000">
  </picture>
</p>

A request passes through five stages: **authenticate** (token to access scope), **retrieve** (semantic, lexical or hybrid, inside that scope), optionally **rerank**, **gate** on evidence strength, then **generate and validate citations**. The REST API and the MCP server share the same services and the same scope.

Key design decisions:

- **Authorization before retrieval, not after.** The access scope is a SQL predicate ANDed into every query, so unauthorized chunks are never fetched. Filtering results afterwards would still let them influence ranking, the evidence gate, a generator or a trace.
- **Refuse rather than guess.** Nothing is generated unless the best chunk clears a similarity threshold, and an answer without a valid citation is refused.
- **Evaluate before claiming quality.** Retrieval changes are measured on a held-out question set that is never used for tuning, with regression floors enforced in CI.
- **No paid dependency on the default path.** Local ONNX embeddings and an extractive generator; LLM generators are optional adapters behind the same gate and citation check.

**Stack:** Python · FastAPI · PostgreSQL · pgvector · HNSW · Hybrid Search · RRF · Cross-Encoder Reranking · LangGraph · MCP · JWT/RBAC · OpenTelemetry · Docker · Terraform · GitHub Actions

## Retrieval modes

Selectable per request (`"mode"` on `/api/search`) or with `RETRIEVAL_MODE`:

| Mode | How it works |
|---|---|
| `semantic` (default) | Embeds the query (`all-MiniLM-L6-v2`, local ONNX) and searches pgvector with an HNSW cosine index; metadata filters run in the same SQL statement |
| `lexical` | PostgreSQL full-text search over a generated `tsvector` column with a GIN index |
| `hybrid` | Runs both and merges the ranked lists with Reciprocal Rank Fusion (`score = Σ 1/(60 + rank)`) |
| `+ rerank` | Optional local cross-encoder re-orders a bounded candidate set from any mode; off by default |

Every mode returns a real cosine similarity, because the evidence gate depends on it. Details: [docs/RETRIEVAL.md](docs/RETRIEVAL.md).

## Grounded answers: evidence gate, generate or refuse, citations

`/api/ask` runs a LangGraph state machine:

```
validate_query → retrieve → assess_evidence → generate_answer | refuse → validate_citations
```

The best chunk's cosine similarity must reach `EVIDENCE_MIN_SIMILARITY` (default `0.55`) before any text is generated; otherwise the question is refused. The generator cites chunks as `[1]`, `[2]`; markers that do not match a supplied chunk are removed, and an answer left with no valid citation is refused. The default `extractive` generator quotes retrieved sentences and needs no model; optional `openai_compatible` and `bedrock` generators receive only chunks that passed the gate and go through the same citation check.

## MCP

A read-only Model Context Protocol server (stdio) with three bounded tools that reuse the same retrieval and RAG services as the REST API: `search_policy`, `get_policy_document` and `ask_policy`. Arguments are typed and length-limited, the tools have no SQL, filesystem, shell or network access, and each call has a deadline. See [docs/OPERATIONS.md](docs/OPERATIONS.md#mcp-server).

## Security & Trust Boundaries

- **Token validation:** with `AUTH_MODE=jwt`, every request carries a bearer token validated for signature, issuer, audience, subject and expiry against an OIDC/JWKS endpoint or a PEM key. The algorithm allowlist is fixed by the key source (asymmetric only for public keys), never taken from the token. A shared secret exists only for local demos, and `AUTH_MODE=off` is a local demo mode.
- **Retrieval-time authorization:** the token's roles, tenant and department become an access scope applied inside the semantic, lexical and hybrid SQL, the document listing and the document reader. Tenants are isolated, the role sets the highest readable access level, and `restricted` and `confidential` documents also require department membership. A document missing an authorization attribute is unreadable (fail closed). Caller filters can narrow the scope but never widen it.
- **Read-only, bounded MCP tools:** three typed, length-limited tools with no SQL, filesystem, shell or network access and a per-call deadline; with JWT on, the server will not start without a valid token, and every call is limited by its scope.
- **Grounding controls:** the evidence gate and citation validation stop unsupported answers; LLM generators receive only chunks that passed the gate.
- **Data-minimized telemetry:** traces, metrics and logs record the question's length, not its text (a short preview only behind an explicit opt-in), and never document text, answers, request bodies, headers or credentials.
- **Secrets:** no secrets files are tracked (checked in CI); in the AWS reference deployment, database credentials come from Secrets Manager and tokens are verified against the identity provider's public keys.

Trade-off: enforcing authorization before retrieval ties the access model to the schema (every chunk carries tenant, department and access level), in exchange for a guarantee that restricted text never reaches ranking, gating, generation or telemetry. This project validates tokens but does not issue them or run an identity provider. Details: [docs/SECURITY.md](docs/SECURITY.md).

## Observability

OpenTelemetry spans for each pipeline stage, Prometheus metrics at `/metrics`, and structured JSON logs with request and trace ids; OTLP export is optional. Question text (only its length is recorded), document text, answers, request headers or bodies and credentials are not recorded. See [docs/OPERATIONS.md](docs/OPERATIONS.md#observability).

## Measured results

Retrieval is evaluated on separate tuning and held-out question sets over the synthetic policy corpus (7 documents, 34 chunks, sentence-transformers/all-MiniLM-L6-v2, pgvector 0.8.7); CI enforces regression floors for both. The corpus is small and synthetic: high scores here do not predict performance on a large corpus.

Held-out set (45 answerable questions, `status=current`):

| Configuration | Hit@1 | Hit@3 | Hit@5 | Recall@5 | nDCG@5 | MRR |
|---|---|---|---|---|---|---|
| semantic (default) | 91.1% | 95.6% | 97.8% | 96.7% | 0.945 | 0.939 |
| lexical | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 |
| hybrid | 97.8% | 100.0% | 100.0% | 100.0% | 0.992 | 0.989 |
| semantic+rerank | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 |
| hybrid+rerank | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 |

Tuning-set results, latency and the evaluation protocol are in [eval/README.md](eval/README.md); raw numbers in [eval/results.json](eval/results.json).

## Testing

490 automated tests: 273 unit tests and 217 integration tests against real PostgreSQL + pgvector, covering retrieval, authorization, the evidence gate, citations and telemetry data minimization. CI runs ruff, mypy, both test suites with retrieval regression floors, Terraform fmt/validate with a Trivy scan, pip-audit, bandit and a Docker Compose smoke test. Commands: [docs/OPERATIONS.md](docs/OPERATIONS.md).

## AWS Reference Deployment

Terraform-defined reference architecture; not currently deployed.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/aws-reference-deployment-dark.svg">
    <img src="docs/aws-reference-deployment.svg" alt="AWS reference deployment defined in Terraform and not currently deployed. Clients on allowed networks reach an Application Load Balancer over HTTPS with an ACM certificate and a TLS 1.3 policy; the load balancer is internal by default and a public one requires a certificate. It forwards to an ECS Fargate service in the application subnets (private when the NAT gateway option is enabled), which alone may connect to RDS PostgreSQL 16 with pgvector in private data subnets. The task pulls its image from ECR, reads database credentials from Secrets Manager, writes to CloudWatch Logs, verifies tokens against the identity provider's JWKS endpoint, and may invoke Bedrock only when model ARNs are granted. IAM execution and task roles are least privilege." width="1000">
  </picture>
</p>

ECS Fargate behind an Application Load Balancer, RDS PostgreSQL 16 + pgvector in private subnets, ECR, Secrets Manager, CloudWatch Logs and least-privilege IAM roles; the task role gets Bedrock access only when model ARNs are listed. `terraform fmt`, `validate` and a Trivy scan run in CI. Modules, network rules and cost notes: [infra/terraform/README.md](infra/terraform/README.md).

## Quick start

Requires Docker. The first build downloads the embedding model into the image.

```bash
docker compose up --build -d                       # PostgreSQL+pgvector and the API on :8000
docker compose exec api python -m scripts.ingest   # embed and store the 7 sample policies
curl -s localhost:8000/api/ask -H 'content-type: application/json' \
  -d '{"question": "What is the late payment fee?"}'
```

A question the documents do not cover, such as *"How do I bake sourdough bread?"*, comes back `refused` with no sources. Tests and configuration: [docs/OPERATIONS.md](docs/OPERATIONS.md); JWT demo tokens: [docs/SECURITY.md](docs/SECURITY.md#local-demo-tokens).

## Documentation

| Document | Contents |
|---|---|
| [docs/RETRIEVAL.md](docs/RETRIEVAL.md) | Schema, chunking, ingestion, retrieval modes, RRF, reranking, HNSW/pgvector tuning, evidence gate, citations |
| [docs/SECURITY.md](docs/SECURITY.md) | JWT/OIDC/JWKS validation, algorithm allowlist, roles, tenant/department/access-level rules, retrieval-time enforcement, MCP security, data minimization |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | API, local environment and tests, configuration reference, generator providers (including Bedrock), MCP server, observability, layout |
| [eval/README.md](eval/README.md) | Evaluation protocol, metrics, tuning and held-out results, latency |
| [infra/terraform/README.md](infra/terraform/README.md) | AWS reference deployment, not currently deployed (ECS Fargate, ALB, RDS PostgreSQL 16, Secrets Manager, IAM) |

## Scope and limitations

A personal project, designed, built and maintained by [@Kalab21](https://github.com/Kalab21).


- The policy corpus is synthetic and small (7 documents, 34 chunks).
- The Terraform AWS configuration is a reference architecture: `terraform fmt`, `validate` and a Trivy scan run in CI, but it has not been applied to an AWS account.
- The AWS Bedrock adapter is covered by tests with the AWS SDK client mocked; live use depends on AWS credentials and model access and is not claimed.
- Only LangGraph is used from the LangChain ecosystem; there is no other LangChain code in the app.
- This project does not issue tokens or run an identity provider; `AUTH_MODE=off` is a local demo mode.
