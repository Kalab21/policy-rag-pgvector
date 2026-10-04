# Policy RAG — pgvector Retrieval Platform

A small retrieval-augmented generation (RAG) service over **synthetic** lending-policy documents. Embeddings live in **PostgreSQL + pgvector**, search is a real vector query with metadata filters, and an **evidence gate** makes the service refuse questions the documents cannot support instead of guessing.

It runs end to end with no paid API: a local embedding model, and an extractive answer generator by default.

```
Query ─► Embedding ─► pgvector HNSW ─► Top-K ─► Evidence gate ─► Answer ─► Citation check
         (MiniLM,     cosine search    chunks    similarity ≥     (extractive   every [n] must point
          384-dim)    + metadata       + scores  threshold?       or LLM)       at a retrieved chunk
                      filter in SQL                │ no
                                                   └──► Refuse (no sources, reason returned)
```

## What is implemented

| Area | Implementation |
|---|---|
| API | FastAPI: `GET /health`, `GET /api/documents`, `POST /api/search`, `POST /api/ask` |
| Vector store | PostgreSQL 16 + pgvector (`pgvector/pgvector:pg16`, 0.8.7 in the runs below), `vector(384)` column |
| Index | HNSW, `vector_cosine_ops`, `m=16`, `ef_construction=64`; `hnsw.ef_search=40` per query |
| Similarity | Cosine distance (`<=>`); similarity = 1 − distance |
| Metadata filtering | In the same SQL statement: `metadata @> filter` (JSONB, GIN-indexed) on `category`, `version`, `status`, `document`, `section` |
| Filtered ANN | pgvector ≥ 0.8 iterative scan (`hnsw.iterative_scan = strict_order`) so a selective filter still returns `top_k` rows |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via [fastembed](https://github.com/qdrant/fastembed) (ONNX, runs locally on CPU), behind an `EmbeddingProvider` interface. Model name, dimension and a content hash are stored per document |
| Ingestion | Markdown with front matter → normalise → section-aware chunking → embed → store. Idempotent (see below) |
| RAG | LangGraph state machine: `validate_query → retrieve → assess_evidence → generate_answer \| refuse → validate_citations` |
| Generation | `extractive` (default, no LLM): quotes the best-matching sentences. Optional `openai_compatible` client for a chat model you configure |
| Evaluation | 33 answerable + 13 unanswerable gold questions; Hit@K, Recall@K, MRR, gate sweep, end-to-end behaviour |
| Tests | 139 (71 unit, 68 integration against real PostgreSQL + pgvector) |
| CI | ruff, mypy, unit + integration tests with a pgvector service container, pip-audit, bandit, Docker Compose smoke test |

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

**Schema.** `documents` (name, title, version, category, status, `content_hash`, `embedding_model`, `embedding_dim`) and `chunks` (`chunk_text`, `chunk_hash`, JSONB `metadata`, `embedding vector(384)`). The API refuses to start if the column width and `EMBEDDING_DIM` disagree.

**Chunking.** Split on markdown headings, then pack whole sentences up to 700 characters with a 120-character overlap; over-long sentences split on words. The text embedded is `"{title} | {section}\n{chunk}"`, while the stored and cited text is the plain chunk. On the bundled documents each section fits in one chunk, so the overlap is never exercised.

**Idempotent ingestion.** A document's fingerprint covers its content, the embedding model and the chunk settings. Running ingest again skips unchanged documents without re-embedding, replaces a changed document atomically, and leaves existing rows untouched if embedding fails.

**Retrieval is separate from RAG.** `app/retrieval` knows nothing about LLMs or the graph; `app/rag` calls it as a service.

**Evidence gate.** Before any text is generated, the best chunk's cosine similarity must reach `EVIDENCE_MIN_SIMILARITY` (default `0.55`). Chunks below it are never given to the generator.

**Citation check.** The generator cites sources as `[1]`, `[2]`. Markers that do not match a supplied chunk are removed, and an answer left with no valid citation is refused.

**Two generators.** `extractive` needs no model: it ranks sentences from the retrieved chunks by embedding similarity to the question and quotes them. It cannot invent text, but it also cannot synthesise across sentences. `openai_compatible` posts to any chat-completions endpoint you configure (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`); it is covered by unit tests against a mocked HTTP transport and **has not been run against a live LLM**.

## Measured results

From `python -m scripts.evaluate_retrieval` on this repository's sample data: 7 documents, 34 chunks, MiniLM-L6-v2 (384 dims), pgvector 0.8.7, extractive generator, threshold 0.55. Raw numbers are in [`eval/results.json`](eval/results.json), questions in [`eval/gold.json`](eval/gold.json).

**Retrieval** (33 answerable questions; a hit is a top-K chunk whose document, version and section match the gold section):

| | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| No status filter (superseded policy can match) | 87.9% | 93.9% | 100.0% | 0.924 |
| `status=current` (what `/api/ask` uses) | 90.9% | 93.9% | 100.0% | 0.939 |

Recall@K equals Hit@K here because each question has exactly one relevant section.

**Evidence gate** (top-1 similarity under current policy):

| Threshold | Answerable questions passing | Unanswerable questions refused |
|---|---|---|
| 0.45 | 97.0% | 84.6% |
| 0.50 | 87.9% | 84.6% |
| **0.55 (default)** | 81.8% | 92.3% |
| 0.60 | 66.7% | 100.0% |

The two score ranges overlap (lowest answerable top-1 similarity 0.430, highest unanswerable 0.582), so no threshold separates them perfectly. The default favours refusing over answering.

**End to end** (`/api/ask` logic): 23 of 33 answerable questions were answered correctly with a relevant citation; 6 were wrongly refused by the gate; 4 were answered with the wrong sentence or source. 12 of 13 unanswerable questions were refused. The one answered was *"What APR does a personal loan carry?"*, which sits close to the fee schedule's mention of APR.

Read these numbers with care:
- **They are in-sample.** The threshold and the extractive sentence ranking were chosen while looking at this same question set, so they are optimistic. There is no held-out set.
- **The corpus is tiny and synthetic** (34 chunks). High Hit@5 on 34 chunks says little about large corpora.
- **The answer check is a strict substring match.** One of the "incorrect" answers (prepayment) is a reasonable answer scored wrong because it says "without additional charges" rather than "no prepayment penalty".
- No latency or throughput benchmarks were run, and none are claimed.

## Scope and limitations

- Synthetic documents only, English only, markdown input only (no PDF/HTML parsing).
- With 34 rows PostgreSQL normally picks an exact scan over the HNSW index. The tests prove the query shape can use the index (`enable_seqscan=off` plus an iterative-scan test), not that the index speeds up this tiny corpus.
- One embedding model, no reranker, no hybrid (keyword + vector) search.
- The default answer generator is extractive, not an LLM. The optional LLM client is untested against a live endpoint.
- The API has no authentication, rate limiting or multi-tenancy. The Compose file uses local-only demo credentials (`policy_rag_local_only`); they are not secrets.
- Not production-ready. It is a portfolio project that demonstrates the retrieval design, the evidence gate and its evaluation.

## Configuration

Environment variables (see [`.env.example`](.env.example)): `DATABASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EVIDENCE_MIN_SIMILARITY`, `RAG_MAX_CONTEXT_CHUNKS`, `LLM_PROVIDER` (`extractive` | `openai_compatible`), `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `HNSW_M`, `HNSW_EF_CONSTRUCTION`, `HNSW_EF_SEARCH`. Changing the embedding model to one with a different width requires recreating the `chunks` table.

## Layout

```
app/api         HTTP routes            app/embeddings  provider interface + fastembed
app/core        settings               app/ingestion   loader, normalise, chunk, pipeline
app/db          schema, pool           app/retrieval   pgvector search + filters
app/models      API + domain types     app/rag         evidence gate, generators, LangGraph
app/evaluation  metrics, gold, runner  scripts         ingest, evaluate_retrieval
sample_data     synthetic policies     eval            gold questions + latest results
tests/unit  tests/integration
```
