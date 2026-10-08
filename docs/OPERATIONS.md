# Operations

Running, configuring and observing the service: API surface, local environment and tests, configuration reference, generator providers, the MCP server and telemetry.

## API

FastAPI: `GET /health`, `GET /metrics`, `GET /api/me`, `GET /api/documents`, `POST /api/search`, `POST /api/ask`. Request and response examples for search and ask are in [RETRIEVAL.md](RETRIEVAL.md#retrieval-modes).

## Local environment

Requires Docker. The first build downloads the embedding model into the image.

```bash
docker compose up --build -d                       # PostgreSQL+pgvector and the API on :8000
docker compose exec api python -m scripts.ingest   # embed and store the 7 sample policies
curl localhost:8000/health
```

### Tests and evaluation

```bash
pip install -r requirements-dev.txt
docker compose up -d db                            # pgvector on localhost:5433
docker compose exec db createdb -U policy_rag policy_rag_test
export TEST_DATABASE_URL=postgresql://policy_rag:policy_rag_local_only@localhost:5433/policy_rag_test
pytest tests/unit                                  # no database needed
pytest tests/integration                           # real PostgreSQL + pgvector
python -m scripts.evaluate_retrieval               # run after ingest; prints the evaluation tables
```

Without `TEST_DATABASE_URL`, integration tests are skipped. They drop and recreate the application tables, so point it at a throwaway database. To evaluate inside the stack instead: `docker compose exec api python -m scripts.evaluate_retrieval`. The evaluation protocol and results are in [`eval/README.md`](../eval/README.md).

| | |
|---|---|
| Tests | 490 (273 unit, 217 integration against real PostgreSQL + pgvector) |
| CI | ruff, mypy, unit + integration tests with a pgvector service container, pip-audit, bandit, Docker Compose smoke test, terraform fmt/validate and a Trivy IaC scan. All six jobs are required checks on `main` |

## Verification status

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

## Configuration

Copy [`.env.example`](../.env.example) to `.env` to override anything; every value in it is the application default. Docker Compose forwards these to the API container:

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

### Configuration notes

- Changing `EMBEDDING_MODEL` needs `docker compose up --build`. If the new model has a different vector width, also set `EMBEDDING_DIM` and recreate the `chunks` table (for example `docker compose down -v`), because the API refuses to start when the column width and `EMBEDDING_DIM` disagree.
- `HNSW_M` and `HNSW_EF_CONSTRUCTION` only apply when the index is created; `HNSW_EF_SEARCH` applies to every query.
- Inside a container `localhost` is the container itself, so point `LLM_BASE_URL` at a reachable host (for example `http://host.docker.internal:11434/v1`).

## Generator providers

`extractive` (default) needs no model: it ranks sentences from the retrieved chunks by embedding similarity to the question and quotes them. Because it quotes retrieved text, it cannot invent content. `openai_compatible` posts to any chat-completions endpoint you configure (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`). `bedrock` calls AWS Bedrock through the official SDK (`boto3`) and the Converse API. Every generator only ever receives chunks that passed the evidence gate, and every answer goes through the same citation check afterwards (see [RETRIEVAL.md](RETRIEVAL.md#citation-check)).

### Bedrock

Set `LLM_PROVIDER=bedrock` and `BEDROCK_MODEL_ID` (there is deliberately no default model: which models an account can use is account-specific), optionally `BEDROCK_REGION`, `BEDROCK_MAX_TOKENS`, `BEDROCK_TEMPERATURE` (default 0), `BEDROCK_TIMEOUT_S` and `BEDROCK_MAX_RETRIES`. Credentials come from the normal AWS chain (environment, profile, or an IAM role) and are never read from this project's settings or committed.

The model is forced to answer through a tool call with a fixed JSON schema (answer, cited source numbers, an explicit "insufficient evidence" flag); the result is validated and the generator **fails closed**: output that is not valid, cites a source that was not supplied, or cites nothing is never served and the question is refused (`generator_output_invalid`). The sources are passed as untrusted reference text, and provider errors are translated to a safe 502 that carries no credentials or request details.

The adapter is implemented through the official AWS SDK and covered by unit and integration tests at the SDK-client level. `BEDROCK_MODEL_ID=<model> BEDROCK_REGION=<region> python -m scripts.bedrock_smoke` makes one request against your own account; live use depends on AWS credentials and model access.

## MCP server

```bash
python -m app.mcp_server                                  # stdio, from the repo with the database reachable
docker compose exec -T api python -m app.mcp_server       # or inside the running stack
```

Point an MCP client at that command. It exposes exactly three read-only tools that reuse the same retrieval and RAG services as the REST API (the logic is not duplicated):

| Tool | Arguments | Returns |
|---|---|---|
| `search_policy` | `query` (1-1000 chars), `top_k` (1-10), optional `filters`, optional `mode` | ranked passages with document, version, section and similarity |
| `get_policy_document` | `document` (slug), optional `version` | one stored document, passage by passage (size-capped) |
| `ask_policy` | `question`, optional `filters` | a cited answer, or a refusal when the evidence is insufficient |

Each call has a deadline (`MCP_TOOL_TIMEOUT_S`). The server runs over stdio, so the client that launches the process supplies its own credential (`MCP_ACCESS_TOKEN` when JWT authentication is on). Argument validation, error masking and scope enforcement are described in [SECURITY.md](SECURITY.md#mcp-security).

## Observability

Enough telemetry to investigate a bad answer without recording what users asked.

- **Traces (OpenTelemetry).** One trace per request: `http.request` → `rag.ask` → `embedding.query`, `retrieval.semantic` / `retrieval.lexical` / `retrieval.hybrid` (with `retrieval.fusion`), `retrieval.rerank`, `evidence.assess`, `generator.generate`, `citation.validate`; MCP calls get an `mcp.tool` span. Attributes are things like retrieval mode, `top_k`, candidate and returned counts, best similarity, the evidence threshold, answer/refusal status and reason, generator and model name, and error *type*.
- **Metrics.** Request, error, ask and refusal counters (refusals by reason); latency histograms for requests, retrieval, reranking and generation; returned-chunk counts; provider-error and MCP tool call/failure counters. Prometheus text is served at `GET /metrics`; there is no collector or dashboard bundled.
- **Logs.** One JSON line per event on stderr, carrying the request id (the `X-Request-ID` header is honoured if it is well-formed, otherwise generated, and returned) plus the trace and span ids, so a log line can be matched to its trace.
- **What is never recorded.** Question text (only its length; a 120-character preview is available behind `RECORD_QUERY_TEXT=true`), document text, answers, request headers or bodies, and credentials. Attribute names that suggest secrets are dropped; credential-shaped strings are masked in logs; span errors carry the exception type, not its message. Tests assert these properties.
- **Export is optional.** Nothing leaves the process unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set (OTLP over HTTP; verified in a test against a local receiver). `TELEMETRY_ENABLED=false` turns it all off.

## Repository layout

```
app/api         HTTP routes            app/embeddings  provider interface + fastembed
app/core        settings               app/ingestion   loader, normalise, chunk, pipeline
app/db          schema, pool           app/retrieval   pgvector search + filters
app/models      API + domain types     app/rag         evidence gate, generators, LangGraph
app/evaluation  metrics, gold, runner  app/security    JWT validation, access scope
app/mcp_server  MCP tools (stdio)      app/observability  traces, metrics, logs
scripts         ingest, evaluate_retrieval, make_demo_token, bedrock_smoke
sample_data     synthetic policies     eval            gold questions + latest results
infra/terraform  AWS reference architecture (Terraform)
tests/unit  tests/integration
```
