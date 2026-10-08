# Policy RAG Platform

**Advanced retrieval and grounded RAG on PostgreSQL + pgvector.**

A retrieval-augmented generation (RAG) service over **synthetic** lending-policy documents. It answers questions only from documents the caller is authorized to read, cites its sources, and refuses when the evidence is insufficient. It runs end to end with no paid API: a local embedding model and an extractive answer generator by default.

## Key capabilities

- **Retrieval:** semantic search (pgvector HNSW), PostgreSQL full-text search and hybrid search with Reciprocal Rank Fusion, plus optional cross-encoder reranking
- **Security:** retrieval-time JWT/RBAC document authorization: with authentication on, restricted documents are never retrieved, ranked or sent to a generator
- **Grounding:** a LangGraph flow with an evidence gate, citation validation and refusal of unsupported questions
- **Interfaces and operations:** an MCP tool server, OpenTelemetry traces, metrics and structured logs
- **Evaluation:** separate tuning and held-out question sets (Hit@K, Recall@K, MRR, nDCG) with regression floors in CI
- **Engineering:** 490 automated tests, 217 of them against real PostgreSQL + pgvector

## Ownership

A personal project, designed, built and maintained by [@Kalab21](https://github.com/Kalab21).

## Architecture

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/architecture-dark.svg">
    <img src="docs/architecture.svg" alt="Policy RAG Platform architecture. A client request passes JWT validation, which produces an access scope applied inside every query. Semantic (pgvector HNSW), lexical (PostgreSQL full-text) or hybrid (Reciprocal Rank Fusion) retrieval produces a candidate set, optionally reranked by a cross-encoder. An evidence gate refuses when similarity is insufficient; otherwise a LangGraph flow generates an answer and citation validation returns it with sources or refuses. MCP, OpenTelemetry, Terraform and CI run across the platform." width="1000">
  </picture>
</p>

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

## Retrieval-time authorization

With `AUTH_MODE=jwt`, every request carries a bearer token validated for signature, issuer, audience, subject and expiry (OIDC/JWKS or PEM key; a shared secret for local demos only), with the algorithm allowlist fixed by the key source, not the token. The token's roles, tenant and department become a SQL predicate ANDed into the semantic, lexical and hybrid queries, so documents the caller may not read are never fetched, ranked, reranked, scored, handed to a generator, returned or logged. Caller filters can narrow that scope but never widen it. The HTTP API and the MCP server obey the same scope. Details: [docs/SECURITY.md](docs/SECURITY.md).

## Grounded answers: evidence gate, generate or refuse, citations

`/api/ask` runs a LangGraph state machine:

```
validate_query → retrieve → assess_evidence → generate_answer | refuse → validate_citations
```

The best chunk's cosine similarity must reach `EVIDENCE_MIN_SIMILARITY` (default `0.55`) before any text is generated; otherwise the question is refused. The generator cites chunks as `[1]`, `[2]`; markers that do not match a supplied chunk are removed, and an answer left with no valid citation is refused. The default `extractive` generator quotes retrieved sentences and needs no model; optional `openai_compatible` and `bedrock` generators receive only chunks that passed the gate and go through the same citation check.

## MCP

A read-only Model Context Protocol server (stdio) with three bounded tools that reuse the same retrieval and RAG services as the HTTP API: `search_policy`, `get_policy_document` and `ask_policy`. Arguments are typed and length-limited, the tools have no SQL, filesystem, shell or network access, and each call has a deadline. See [docs/OPERATIONS.md](docs/OPERATIONS.md#mcp-server).

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
| [infra/terraform/README.md](infra/terraform/README.md) | AWS reference architecture (ECS Fargate, ALB, RDS PostgreSQL 16, Secrets Manager, IAM) |

## Scope and limitations

- The policy corpus is synthetic and small (7 documents, 34 chunks).
- The Terraform AWS configuration is a reference architecture: `terraform fmt`, `validate` and a Trivy scan run in CI, but it has not been applied to an AWS account.
- The AWS Bedrock adapter is covered by tests with the AWS SDK client mocked; live use depends on AWS credentials and model access and is not claimed.
- Only LangGraph is used from the LangChain ecosystem; there is no other LangChain code in the app.
- This project does not issue tokens or run an identity provider; `AUTH_MODE=off` is a local demo mode.
