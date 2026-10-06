# Policy RAG Platform

**Advanced retrieval and grounded RAG on PostgreSQL + pgvector.**

A retrieval-augmented generation (RAG) service over **synthetic** lending-policy documents. It runs end to end with no paid API: a local embedding model and an extractive answer generator by default.

- **Retrieval:** semantic search (pgvector HNSW), PostgreSQL full-text search and hybrid search with Reciprocal Rank Fusion, plus optional cross-encoder reranking
- **Security:** retrieval-time JWT/RBAC document authorization: with authentication on, restricted documents are never retrieved, ranked or sent to a generator
- **Grounding:** a LangGraph flow with an evidence gate, citation validation and refusal of unsupported questions
- **Interfaces and operations:** an MCP tool server, OpenTelemetry traces, metrics and structured logs
- **Evaluation:** separate tuning and held-out question sets (Hit@K, Recall@K, MRR, nDCG) with regression floors in CI
- **Engineering:** 490 automated tests, 217 of them against real PostgreSQL + pgvector

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/architecture-dark.svg">
    <img src="docs/architecture.svg" alt="Policy RAG Platform architecture. A client request passes JWT validation, which produces an access scope applied inside every query. Semantic (pgvector HNSW), lexical (PostgreSQL full-text) or hybrid (Reciprocal Rank Fusion) retrieval produces a candidate set, optionally reranked by a cross-encoder. An evidence gate refuses when similarity is insufficient; otherwise a LangGraph flow generates an answer and citation validation returns it with sources or refuses. MCP, OpenTelemetry, Terraform and CI run across the platform." width="1000">
  </picture>
</p>

**Stack:** Python · FastAPI · PostgreSQL · pgvector · HNSW · Hybrid Search · RRF · Cross-Encoder Reranking · LangGraph · MCP · JWT/RBAC · OpenTelemetry · Docker · Terraform · GitHub Actions

## Status at a glance

| Capability | Status |
|---|---|
| pgvector semantic search, HNSW, metadata filtering | Verified (real PostgreSQL + pgvector in tests and CI) |
| PostgreSQL full-text search, hybrid retrieval, RRF | Verified; selectable per request |
| Cross-encoder reranking | Verified with a real local model; optional |
| Held-out retrieval evaluation, nDCG, CI floors | Verified |
| LangGraph evidence gate, citations, refusals | Verified |
| JWT authentication, roles, document-level authorization | Implemented and tested end to end (signed tokens, JWKS discovery) |
| MCP server | Implemented and tested (in-process and stdio clients) |
| OpenTelemetry, Prometheus metrics, structured logs | Implemented and tested, including OTLP export |
| AWS Bedrock adapter | Implemented through the official AWS SDK and covered by adapter tests; live use depends on account and model access |
| Terraform AWS reference architecture | ECS/Fargate, ALB, RDS PostgreSQL, Secrets Manager and IAM; fmt, validate and Trivy verified in CI |

## What is implemented

| Area | Implementation |
|---|---|
| API | FastAPI: `GET /health`, `GET /metrics`, `GET /api/me`, `GET /api/documents`, `POST /api/search`, `POST /api/ask` |
| Vector store | PostgreSQL 16 + pgvector (`pgvector/pgvector:pg16`, 0.8.7 in the runs below), `vector(384)` column |
| Index | HNSW, `vector_cosine_ops`, `m=16`, `ef_construction=64`; `hnsw.ef_search=40` per query |
| Similarity | Cosine distance (`<=>`); similarity = 1 − distance |
| Retrieval modes | `semantic` (pgvector, default), `lexical` (PostgreSQL full-text), `hybrid` (both, fused with Reciprocal Rank Fusion), selectable per request |
| Reranking | Optional local cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2` via fastembed, ONNX) re-orders a candidate set from any retrieval mode; off by default |
| Metadata filtering | In the same SQL statement: `metadata @> filter` (JSONB, GIN-indexed) on `category`, `version`, `status`, `document`, `section` |
| Filtered ANN | pgvector ≥ 0.8 iterative scan (`hnsw.iterative_scan = strict_order`) lets HNSW keep scanning for matching rows after the filter, which helps fill `top_k` when at least `top_k` matching rows exist |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via [fastembed](https://github.com/qdrant/fastembed) (ONNX, runs locally on CPU), behind an `EmbeddingProvider` interface. Model name, dimension and a content hash are stored per document |
| Ingestion | Markdown with front matter → normalise → section-aware chunking → embed → store. Idempotent (see below) |
| RAG | LangGraph state machine: `validate_query → retrieve → assess_evidence → generate_answer \| refuse → validate_citations` |
| Generation | `extractive` (default, no LLM): quotes the best-matching sentences. Optional `bedrock` (AWS Bedrock Converse API) and `openai_compatible` generators for a chat model you configure; the Bedrock adapter is covered by SDK-level tests |
| Evaluation | Separate tuning (33 + 13) and held-out (45 + 16) question sets; Hit@K, Recall@K, nDCG@K, MRR, per-configuration latency, gate sweep, end-to-end behaviour; CI floors on both |
| MCP | A read-only Model Context Protocol server (stdio) with three bounded tools: `search_policy`, `get_policy_document`, `ask_policy` |
| Observability | OpenTelemetry spans for each pipeline stage, Prometheus metrics at `/metrics`, structured JSON logs with request ids, optional OTLP export |
| Security | JWT validation (OIDC/JWKS, PEM key, or demo shared secret), roles, and document-level authorization enforced inside the retrieval SQL (tenant, access level, department); the HTTP API and the MCP server obey the same scope |
| Infrastructure | Terraform reference architecture for AWS (ECS Fargate, ALB, RDS PostgreSQL 16, Secrets Manager, CloudWatch, least-privilege IAM). `terraform fmt`, `validate` and a Trivy scan run in CI (see [`infra/terraform`](infra/terraform/README.md)) |
| Tests | 490 (273 unit, 217 integration against real PostgreSQL + pgvector) |
| CI | ruff, mypy, unit + integration tests with a pgvector service container, pip-audit, bandit, Docker Compose smoke test, terraform fmt/validate and a Trivy IaC scan. All six jobs are required checks on `main` |

Only LangGraph is used from the LangChain ecosystem; there is no other LangChain code in the app.

## Quick start

Requires Docker. The first build downloads the embedding model into the image.

```bash
docker compose up --build -d                       # PostgreSQL+pgvector and the API on :8000
docker compose exec api python -m scripts.ingest   # embed and store the 7 sample policies
curl localhost:8000/health
```

### Search

```bash
curl -s localhost:8000/api/search -H 'content-type: application/json' -d '{
  "query": "What is the late payment fee?",
  "top_k": 3,
  "filters": {"category": "fees"}
}'
```

Each result has the chunk `text`, its source (`document`, `title`, `version`, `section`), `category`/`status`, `distance`, and `similarity`.

### Ask (grounded answer or refusal)

```bash
curl -s localhost:8000/api/ask -H 'content-type: application/json' \
  -d '{"question": "What is the late payment fee?"}'
```

Returns `answer`, `status` (`answered` or `refused`), `sources` (the cited chunks), `retrieved_chunk_ids`, and `evidence` (status, reason, best similarity, threshold). A question the documents do not cover, such as *"How do I bake sourdough bread?"*, comes back `refused` with no sources. Answers use **current** policy unless `filters.status` says otherwise (`"superseded"` returns the older underwriting rules).

### MCP server (tools for AI clients)

```bash
python -m app.mcp_server                                  # stdio, from the repo with the database reachable
docker compose exec -T api python -m app.mcp_server       # or inside the running stack
```

Point an MCP client at that command. It exposes exactly three read-only tools that reuse the same retrieval and RAG services as the HTTP API (the logic is not duplicated):

| Tool | Arguments | Returns |
|---|---|---|
| `search_policy` | `query` (1-1000 chars), `top_k` (1-10), optional `filters`, optional `mode` | ranked passages with document, version, section and similarity |
| `get_policy_document` | `document` (slug), optional `version` | one stored document, passage by passage (size-capped) |
| `ask_policy` | `question`, optional `filters` | a cited answer, or a refusal when the evidence is insufficient |

The tools take typed, length-limited arguments (a document name must match `[a-z0-9-]`, so paths and SQL are rejected before any code runs), have no SQL, filesystem, shell or network access, and are annotated read-only. Failures come back as MCP tool errors with a safe message; unexpected exceptions are masked; each call has a deadline (`MCP_TOOL_TIMEOUT_S`). The server can be started with fixed metadata filters that a caller can narrow but never change, and the same scope applies as for the HTTP API. It runs over stdio, so the client that launches the process supplies its own credential (`MCP_ACCESS_TOKEN` when JWT authentication is on).

### Security: authentication and document authorization

`AUTH_MODE=jwt` makes the API and the MCP server validate a bearer token and limit everything to what it allows. The default, `AUTH_MODE=off`, is the local demo mode: no login, and `/api/me` reports it.

**Who may read what.** Every document (and each of its chunks) is labelled with a `tenant_id`, an `access_level` (`public` < `internal` < `restricted` < `confidential`) and a `department`, set in the document's front matter (unlabelled documents are `internal` in the `default` tenant). A token's roles set the highest level its holder may read: `employee` internal, `underwriter` and `compliance` restricted, `admin` confidential. Public and internal documents are open to the whole tenant; restricted and confidential ones also require the holder's department to match (admins belong to every department). Other tenants' documents are never readable, and a chunk with missing labels is treated as unreadable.

**Where it is enforced.** The scope is built only from the validated token and turned into a SQL predicate that is ANDed into the semantic, lexical and hybrid queries, the document listing and the document reader. Restricted chunks are therefore never fetched, so they cannot be ranked, reranked, scored by the evidence gate, handed to a generator, put in a response, or written to a trace or log. Caller metadata filters are a separate, additional condition: they can narrow results but cannot widen the scope, `tenant_id` and the other authorization fields are not accepted as filters, and nothing in the request (query string, headers, body) can name a tenant.

**Token validation.** Signature, issuer, audience, subject and expiry are required, with a small clock leeway. The algorithm allowlist comes from the configured key source, not from the token: a JWKS URL or PEM public key accepts only RS256/ES256, a shared secret (at least 32 characters, local demos only) only HS256, so `alg: none` and HS/RS key-confusion tokens are rejected. Missing or invalid tokens get `401`; a valid token with no recognised role gets `403`. Roles in a token that this project does not define are ignored. Misconfiguration (no key source, a short secret, missing issuer or audience) stops the server at startup rather than running it open.

**MCP.** `python -m app.mcp_server` with `AUTH_MODE=jwt` refuses to start without a valid `MCP_ACCESS_TOKEN` in its environment, and then every tool call is limited by that token's scope.

**Local demo tokens.** `python -m scripts.make_demo_token --role underwriter --department underwriting` mints a short-lived HS256 token from `AUTH_JWT_SECRET`, for demos and tests only. This project does not issue tokens or run an identity provider.

```bash
export AUTH_MODE=jwt AUTH_ISSUER=demo AUTH_AUDIENCE=policy-rag AUTH_JWT_SECRET="$(openssl rand -hex 32)"
docker compose up -d --build && docker compose exec -T api python -m scripts.ingest
TOKEN=$(docker compose exec -T api python -m scripts.make_demo_token --role employee 2>/dev/null)
curl -s localhost:8000/api/me -H "Authorization: Bearer $TOKEN"
```

### AWS reference architecture

[`infra/terraform`](infra/terraform/README.md) describes how this service could run on AWS: Fargate behind an internal-by-default load balancer, RDS PostgreSQL 16 with pgvector in private subnets (its password generated and held by Secrets Manager, never in the configuration), mandatory JWT authentication against your OIDC provider, and narrowly scoped IAM roles. `terraform fmt`, `terraform validate` and a Trivy scan (no HIGH or CRITICAL findings) run in CI.

### Observability

Enough telemetry to investigate a bad answer without recording what users asked.

- **Traces (OpenTelemetry).** One trace per request: `http.request` → `rag.ask` → `embedding.query`, `retrieval.semantic` / `retrieval.lexical` / `retrieval.hybrid` (with `retrieval.fusion`), `retrieval.rerank`, `evidence.assess`, `generator.generate`, `citation.validate`; MCP calls get an `mcp.tool` span. Attributes are things like retrieval mode, `top_k`, candidate and returned counts, best similarity, the evidence threshold, answer/refusal status and reason, generator and model name, and error *type*.
- **Metrics.** Request, error, ask and refusal counters (refusals by reason); latency histograms for requests, retrieval, reranking and generation; returned-chunk counts; provider-error and MCP tool call/failure counters. Prometheus text is served at `GET /metrics`; there is no collector or dashboard bundled.
- **Logs.** One JSON line per event on stderr, carrying the request id (the `X-Request-ID` header is honoured if it is well-formed, otherwise generated, and returned) plus the trace and span ids, so a log line can be matched to its trace.
- **What is never recorded.** Question text (only its length; a 120-character preview is available behind `RECORD_QUERY_TEXT=true`), document text, answers, request headers or bodies, and credentials. Attribute names that suggest secrets are dropped; credential-shaped strings are masked in logs; span errors carry the exception type, not its message. Tests assert these properties.
- **Export is optional.** Nothing leaves the process unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set (OTLP over HTTP; verified in a test against a local receiver). `TELEMETRY_ENABLED=false` turns it all off.

### Tests and evaluation

```bash
pip install -r requirements-dev.txt
docker compose up -d db                            # pgvector on localhost:5433
docker compose exec db createdb -U policy_rag policy_rag_test
export TEST_DATABASE_URL=postgresql://policy_rag:policy_rag_local_only@localhost:5433/policy_rag_test
pytest tests/unit                                  # no database needed
pytest tests/integration                           # real PostgreSQL + pgvector
python -m scripts.evaluate_retrieval               # run after ingest; prints the tables below
```

Without `TEST_DATABASE_URL`, integration tests are skipped. They drop and recreate the application tables, so point it at a throwaway database. To evaluate inside the stack instead: `docker compose exec api python -m scripts.evaluate_retrieval`.

## How it works

**Schema.** `documents` (name, title, version, category, status, `content_hash`, `embedding_model`, `embedding_dim`) and `chunks` (`chunk_text`, `chunk_hash`, JSONB `metadata`, `embedding vector(384)`). The API refuses to start if the column width and `EMBEDDING_DIM` disagree. On the 34-chunk sample PostgreSQL may choose an exact scan; use of the HNSW index is verified with `enable_seqscan=off`.

**Chunking.** Split on markdown headings, then pack whole sentences up to 700 characters with a 120-character overlap; over-long sentences split on words. The text embedded is `"{title} | {section}\n{chunk}"`, while the stored and cited text is the plain chunk. On the bundled documents each section fits in one chunk, so the overlap is never exercised.

**Idempotent ingestion.** A document's fingerprint covers its content, the embedding model and the chunk settings. Running ingest again skips unchanged documents without re-embedding, replaces a changed document atomically, and leaves existing rows untouched if embedding fails.

**Retrieval modes.** `semantic` embeds the query and searches pgvector. `lexical` uses a PostgreSQL full-text index (a generated `tsvector` column with a GIN index, English stemming, stop words dropped, any content word may match). `hybrid` runs both under the same metadata filter and merges the two ranked lists with Reciprocal Rank Fusion (`score = Σ 1/(60 + rank)`), which uses ranks only, so cosine similarities and text-search ranks never have to be compared. Why bother: dense embeddings capture meaning but can rank a chunk containing an exact identifier or rare term below vaguely related text; an integration test builds that situation with hand-made vectors, and hybrid retrieval surfaces the chunk. Choose a mode with `RETRIEVAL_MODE` or `"mode"` in a `/api/search` request. Results from every mode carry a real cosine similarity, because the evidence gate depends on it.

**Reranking.** The retrievers are fast because they score the query and each chunk independently. A cross-encoder reads the query and one chunk together, which is more accurate but too slow for the whole corpus, so it only re-orders a candidate set: the retriever returns `RERANK_CANDIDATES` chunks (never fewer than `top_k`), the cross-encoder scores them, and the best `top_k` are kept. Metadata filters are applied during candidate retrieval, so a filtered-out chunk is never scored. Reranked results keep their cosine similarity, so the evidence gate behaves as before. Enable it with `RERANK_ENABLED=true` or `"rerank": true` on a `/api/search` request.

**Retrieval is separate from RAG.** `app/retrieval` knows nothing about LLMs or the graph; `app/rag` calls it as a service.

**Evidence gate.** Before any text is generated, the best chunk's cosine similarity must reach `EVIDENCE_MIN_SIMILARITY` (default `0.55`). Chunks below it are never given to the generator.

**Citation check.** The generator cites sources as `[1]`, `[2]`. Markers that do not match a supplied chunk are removed, and an answer left with no valid citation is refused.

**Three generators.** `extractive` (default) needs no model: it ranks sentences from the retrieved chunks by embedding similarity to the question and quotes them. Because it quotes retrieved text, it cannot invent content. `openai_compatible` posts to any chat-completions endpoint you configure (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`). `bedrock` calls AWS Bedrock through the official SDK (`boto3`) and the Converse API. Every generator only ever receives chunks that passed the evidence gate, and every answer goes through the same citation check afterwards.

**Bedrock details.** Set `LLM_PROVIDER=bedrock` and `BEDROCK_MODEL_ID` (there is deliberately no default model: which models an account can use is account-specific), optionally `BEDROCK_REGION`, `BEDROCK_MAX_TOKENS`, `BEDROCK_TEMPERATURE` (default 0), `BEDROCK_TIMEOUT_S` and `BEDROCK_MAX_RETRIES`. Credentials come from the normal AWS chain (environment, profile, or an IAM role) and are never read from this project's settings or committed. The model is forced to answer through a tool call with a fixed JSON schema (answer, cited source numbers, an explicit "insufficient evidence" flag); the result is validated and the generator **fails closed**: output that is not valid, cites a source that was not supplied, or cites nothing is never served and the question is refused (`generator_output_invalid`). The sources are passed as untrusted reference text, and provider errors are translated to a safe 502 that carries no credentials or request details.

**Bedrock adapter.** Implemented through the official AWS SDK and covered by unit and integration tests at the SDK-client level. `BEDROCK_MODEL_ID=<model> BEDROCK_REGION=<region> python -m scripts.bedrock_smoke` makes one request against your own account; live use depends on AWS credentials and model access.

## Measured results

Retrieval is evaluated on separate tuning and held-out question sets over the synthetic policy corpus (7 documents, 34 chunks, sentence-transformers/all-MiniLM-L6-v2, pgvector 0.8.7); CI enforces regression floors for both. The protocol is in [`eval/README.md`](eval/README.md) and the raw numbers are in [`eval/results.json`](eval/results.json).

### Held-out set (45 answerable questions, `status=current`)

| Configuration | Hit@1 | Hit@3 | Hit@5 | Recall@5 | nDCG@5 | MRR |
|---|---|---|---|---|---|---|
| semantic (default) | 91.1% | 95.6% | 97.8% | 96.7% | 0.945 | 0.939 |
| lexical | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 |
| hybrid | 97.8% | 100.0% | 100.0% | 100.0% | 0.992 | 0.989 |
| semantic+rerank | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 |
| hybrid+rerank | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 |

### Tuning set (33 answerable questions, `status=current`)

| Configuration | Hit@1 | Hit@3 | Hit@5 | Recall@5 | nDCG@5 | MRR |
|---|---|---|---|---|---|---|
| semantic (default) | 90.9% | 93.9% | 100.0% | 100.0% | 0.954 | 0.939 |
| lexical | 84.8% | 97.0% | 100.0% | 100.0% | 0.930 | 0.907 |
| hybrid | 84.8% | 97.0% | 100.0% | 100.0% | 0.934 | 0.912 |
| semantic+rerank | 97.0% | 97.0% | 100.0% | 100.0% | 0.983 | 0.977 |
| hybrid+rerank | 97.0% | 97.0% | 100.0% | 100.0% | 0.983 | 0.977 |

Semantic retrieval is the default; lexical, hybrid and reranked configurations are selectable per request or through configuration. Reranking runs a local cross-encoder over a candidate set. Per-search latency from the harness (held-out run, milliseconds, one development machine):

| Configuration | p50 | p95 |
|---|---|---|
| semantic | 7 | 9 |
| lexical | 6 | 8 |
| hybrid | 10 | 12 |
| semantic+rerank | 332 | 468 |
| hybrid+rerank | 328 | 384 |

The evidence gate is evaluated separately from retrieval and is covered by regression tests across answerable and unanswerable questions.

## Configuration

Copy [`.env.example`](.env.example) to `.env` to override anything; every value in it is the application default. Docker Compose forwards these to the API container:

| Group | Variables |
|---|---|
| PostgreSQL | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST_PORT` (Compose builds `DATABASE_URL` from them) |
| Embeddings | `EMBEDDING_MODEL` (also a build argument, because the image downloads the model at build time), `EMBEDDING_DIM` |
| Chunking | `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| Retrieval | `RETRIEVAL_MODE`, `RRF_K`, `HYBRID_CANDIDATES`, `RERANK_ENABLED`, `RERANK_MODEL` (also a build argument, because the image downloads it), `RERANK_CANDIDATES`, `HNSW_M`, `HNSW_EF_CONSTRUCTION`, `HNSW_EF_SEARCH` |
| Evidence gate / RAG | `EVIDENCE_MIN_SIMILARITY`, `RAG_MAX_CONTEXT_CHUNKS` |
| Authentication | `AUTH_MODE` (`off` default, or `jwt`), `AUTH_ISSUER`, `AUTH_AUDIENCE`, one of `AUTH_JWKS_URL` / `AUTH_PUBLIC_KEY` / `AUTH_JWT_SECRET`, `AUTH_ROLES_CLAIM`, `AUTH_TENANT_CLAIM`, `AUTH_DEPARTMENT_CLAIM`, `AUTH_LEEWAY_S`; `MCP_ACCESS_TOKEN` for the MCP server |
| MCP | `MCP_TOOL_TIMEOUT_S` |
| Observability | `TELEMETRY_ENABLED`, `METRICS_ENABLED`, `OTEL_SERVICE_NAME`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `RECORD_QUERY_TEXT`, `LOG_FORMAT`, `LOG_LEVEL` |
| Optional LLM | `LLM_PROVIDER` (`extractive` default, `openai_compatible` or `bedrock`), `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` |
| Bedrock | `BEDROCK_MODEL_ID`, `BEDROCK_REGION`, `BEDROCK_MAX_TOKENS`, `BEDROCK_TEMPERATURE`, `BEDROCK_TIMEOUT_S`, `BEDROCK_MAX_RETRIES` (AWS credentials are never configured here) |

Settings that exist in the application but are deliberately not forwarded by Compose: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` and `DB_SSLMODE` (the database in parts, used by the AWS deployment, where a managed secret supplies the password; setting `DB_HOST` replaces `DATABASE_URL`), `DATABASE_URL` (derived from the `POSTGRES_*` values), `EMBEDDING_PROVIDER` (only `fastembed` is implemented), `SAMPLE_DATA_DIR`, `AUTO_INIT_SCHEMA`, `DB_POOL_MIN_SIZE`, `DB_POOL_MAX_SIZE` and `LLM_TIMEOUT_S`. They work when running the app directly, not through Compose.

Notes:
- Changing `EMBEDDING_MODEL` needs `docker compose up --build`. If the new model has a different vector width, also set `EMBEDDING_DIM` and recreate the `chunks` table (for example `docker compose down -v`), because the API refuses to start when the column width and `EMBEDDING_DIM` disagree.
- `HNSW_M` and `HNSW_EF_CONSTRUCTION` only apply when the index is created; `HNSW_EF_SEARCH` applies to every query.
- Inside a container `localhost` is the container itself, so point `LLM_BASE_URL` at a reachable host (for example `http://host.docker.internal:11434/v1`).

## Layout

```
app/api         HTTP routes            app/embeddings  provider interface + fastembed
app/core        settings               app/ingestion   loader, normalise, chunk, pipeline
app/db          schema, pool           app/retrieval   pgvector search + filters
app/models      API + domain types     app/rag         evidence gate, generators, LangGraph
app/evaluation  metrics, gold, runner  scripts         ingest, evaluate_retrieval
sample_data     synthetic policies     eval            gold questions + latest results
infra/terraform  AWS reference architecture (Terraform)
tests/unit  tests/integration
```
