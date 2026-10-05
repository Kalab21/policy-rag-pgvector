# Policy RAG Platform

**Grounded RAG with semantic and optional hybrid retrieval, built on PostgreSQL + pgvector.**

A small retrieval-augmented generation (RAG) service over **synthetic** lending-policy documents. Embeddings live in **PostgreSQL + pgvector**, search is a real vector query with metadata filters, and an **evidence gate** makes the service refuse questions the documents cannot support instead of guessing.

It runs end to end with no paid API: a local embedding model, and an extractive answer generator by default.

```
Query ─► Embedding ─► pgvector HNSW ─► Top-K ─► Evidence gate ─► Answer ─► Citation check
         (MiniLM,     cosine search    chunks    similarity ≥     (extractive   every [n] must point
          384-dim)    + metadata       + scores  threshold?       or LLM)       at a retrieved chunk
                      filter in SQL                │ no
                                                   └──► Refuse (no sources, reason returned)
```

Demonstrates real vector storage, semantic retrieval, grounded answer/refusal logic, and retrieval evaluation using PostgreSQL + pgvector.

**Stack:** Python · FastAPI · PostgreSQL · pgvector (HNSW, cosine similarity) · embeddings · metadata filtering · LangGraph · evidence gating · citation validation · retrieval evaluation · Docker · GitHub Actions · Pytest

## What is implemented

| Area | Implementation |
|---|---|
| API | FastAPI: `GET /health`, `GET /api/documents`, `POST /api/search`, `POST /api/ask` |
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
| Generation | `extractive` (default, no LLM): quotes the best-matching sentences. Optional `openai_compatible` client for a chat model you configure |
| Evaluation | Separate tuning (33 + 13) and held-out (45 + 16) question sets; Hit@K, Recall@K, nDCG@K, MRR, per-configuration latency, gate sweep, end-to-end behaviour; CI floors on both |
| Tests | 218 (97 unit, 121 integration against real PostgreSQL + pgvector) |
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

**Retrieval modes.** `semantic` embeds the query and searches pgvector. `lexical` uses a PostgreSQL full-text index (a generated `tsvector` column with a GIN index, English stemming, stop words dropped, any content word may match). `hybrid` runs both under the same metadata filter and merges the two ranked lists with Reciprocal Rank Fusion (`score = Σ 1/(60 + rank)`), which uses ranks only, so cosine similarities and text-search ranks never have to be compared. Why bother: dense embeddings capture meaning but can rank a chunk containing an exact identifier or rare term below vaguely related text; an integration test builds that situation with hand-made vectors, and hybrid retrieval surfaces the chunk. Choose a mode with `RETRIEVAL_MODE` or `"mode"` in a `/api/search` request. Results from every mode carry a real cosine similarity, because the evidence gate depends on it.

**Reranking.** The retrievers are fast because they score the query and each chunk independently. A cross-encoder reads the query and one chunk together, which is more accurate but too slow for the whole corpus, so it only re-orders a candidate set: the retriever returns `RERANK_CANDIDATES` chunks (never fewer than `top_k`), the cross-encoder scores them, and the best `top_k` are kept. Metadata filters are applied during candidate retrieval, so a filtered-out chunk is never scored. Reranked results keep their cosine similarity, so the evidence gate behaves as before. Enable it with `RERANK_ENABLED=true` or `"rerank": true` on a `/api/search` request.

**Retrieval is separate from RAG.** `app/retrieval` knows nothing about LLMs or the graph; `app/rag` calls it as a service.

**Evidence gate.** Before any text is generated, the best chunk's cosine similarity must reach `EVIDENCE_MIN_SIMILARITY` (default `0.55`). Chunks below it are never given to the generator.

**Citation check.** The generator cites sources as `[1]`, `[2]`. Markers that do not match a supplied chunk are removed, and an answer left with no valid citation is refused.

**Two generators.** `extractive` needs no model: it ranks sentences from the retrieved chunks by embedding similarity to the question and quotes them. It cannot invent text, but it also cannot synthesise across sentences. `openai_compatible` posts to any chat-completions endpoint you configure (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`); it is covered by unit tests against a mocked HTTP transport and **has not been run against a live LLM**.

## Measured results

From `python -m scripts.evaluate_retrieval` on this repository's sample data: 7 synthetic documents, 34 chunks, MiniLM-L6-v2 (384 dims), pgvector 0.8.7, extractive generator, evidence threshold 0.55. Raw numbers: [`eval/results.json`](eval/results.json). The protocol, including what each set may and may not be used for, is in [`eval/README.md`](eval/README.md).

Two question sets are kept apart:

- **Tuning set** (33 answerable + 13 unanswerable questions): used while building the system. The evidence threshold, the extractive sentence ranking and the default retrieval mode were chosen while looking at it.
- **Held-out set** (45 answerable + 16 unanswerable questions): written after the pipeline was fixed and **not used to choose any setting**. It is the number to trust more. It is still small, written by the same person who built the system, and drawn from the same seven documents.

### Retrieval, held-out set (45 answerable questions, `status=current`)

| Configuration | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | nDCG@1 | nDCG@3 | nDCG@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| semantic (default) | 91.1% | 95.6% | 97.8% | 88.9% | 95.6% | 96.7% | 0.911 | 0.939 | 0.945 | 0.939 |
| lexical | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% | 100.0% | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid | 97.8% | 100.0% | 100.0% | 94.4% | 100.0% | 100.0% | 0.978 | 0.992 | 0.992 | 0.989 |
| semantic+rerank | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% | 100.0% | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid+rerank | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% | 100.0% | 1.000 | 1.000 | 1.000 | 1.000 |

### Retrieval, tuning set (33 answerable questions, `status=current`)

| Configuration | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | nDCG@1 | nDCG@3 | nDCG@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| semantic (default) | 90.9% | 93.9% | 100.0% | 90.9% | 93.9% | 100.0% | 0.909 | 0.928 | 0.954 | 0.939 |
| lexical | 84.8% | 97.0% | 100.0% | 84.8% | 97.0% | 100.0% | 0.848 | 0.917 | 0.930 | 0.907 |
| hybrid | 84.8% | 97.0% | 100.0% | 84.8% | 97.0% | 100.0% | 0.848 | 0.921 | 0.934 | 0.912 |
| semantic+rerank | 97.0% | 97.0% | 100.0% | 97.0% | 97.0% | 100.0% | 0.970 | 0.970 | 0.983 | 0.977 |
| hybrid+rerank | 97.0% | 97.0% | 100.0% | 97.0% | 97.0% | 100.0% | 0.970 | 0.970 | 0.983 | 0.977 |

Recall@K is below Hit@K on the held-out set only because three of its questions need two sections.

### What the two sets say, and what was decided

- **On the tuning set, semantic search led the unreranked modes** (MRR 0.939 vs 0.912 hybrid and 0.907 lexical), so `semantic` is the default.
- **On the held-out set the order reversed:** lexical (1.000) and hybrid (0.989) beat semantic (0.939). The held-out set has more short, specific-term questions (for example "adverse action notice"), which is where full-text matching helps. This is a hypothesis for the next evaluation round, not a reason to change the default now: under the protocol the held-out set cannot be used to choose a retrieval mode, and with 45 questions the gap is a handful of questions.
- **Cross-encoder reranking helped on both sets** (held-out MRR 1.000, tuning 0.977) but it stays **off by default**: it adds roughly 300 ms per search on this CPU (below), which fails the 250 ms criterion fixed before the held-out run.
- With only 34 chunks, a 20-candidate rerank depth covers over half the corpus, and held-out scores near 1.000 leave no headroom to see differences. Neither says much about a large corpus.

### Latency (held-out run, milliseconds per search, this machine)

| Configuration | mean | p50 | p95 |
|---|---|---|---|
| semantic | 7 | 7 | 9 |
| lexical | 6 | 6 | 8 |
| hybrid | 10 | 10 | 12 |
| semantic+rerank | 342 | 332 | 468 |
| hybrid+rerank | 318 | 328 | 384 |

One process, a handful of queries, a CPU, a 34-chunk corpus. Indicative of the *relative* cost of the cross-encoder, not a benchmark and not a production number.

### Evidence gate and answering

Gate sweep on the **held-out** set (top-1 cosine similarity, `status=current`):

| Threshold | Answerable passing | Unanswerable refused |
|---|---|---|
| 0.45 | 91.1% | 81.2% |
| 0.50 | 84.4% | 81.2% |
| **0.55 (default)** | 71.1% | 87.5% |
| 0.60 | 51.1% | 93.8% |

Lowest answerable top-1 similarity 0.181; highest unanswerable 0.731. The ranges overlap heavily, so no single threshold separates them. The tuning set looked better (0.430 / 0.582) because the threshold was chosen on it.

End to end (`/api/ask` logic, default settings):

| | Tuning | Held-out |
|---|---|---|
| Answerable answered correctly with a relevant citation | 23/33 | 28/45 |
| Answerable wrongly refused by the gate | 6 | 13 |
| Answerable answered with a wrong sentence or source | 4 | 4 |
| Unanswerable refused | 12/13 | 14/16 |

The gate is the weakest part: on held-out questions it refused 13 of 45 answerable ones, and it answered 2 unanswerable near-domain questions (an auto-loan limit and a 36-month interest rate, both topics the documents touch without answering). A better gate would need more than a single similarity threshold, and was not attempted here.

### Reading these numbers

- The held-out set is separate from the tuning set but small (n = 45) and written by the author of the system.
- The corpus is tiny and synthetic. High Hit@5 on 34 chunks says little about large corpora.
- The answer check is a strict substring match, so a reasonable answer phrased differently can be scored wrong.
- Latency figures are single-machine and indicative only. No throughput benchmark was run.
- CI enforces floors below these measured values on both sets (`tests/integration/test_evaluation_regression.py`). They are regression guards, not quality claims.

## Scope and limitations

- Synthetic documents only, English only, markdown input only (no PDF/HTML parsing).
- With 34 rows PostgreSQL normally picks an exact scan over the HNSW index. The tests prove the query shape can use the index (`enable_seqscan=off` plus an iterative-scan test), not that the index speeds up this tiny corpus.
- One embedding model. Hybrid search and cross-encoder reranking are optional and off by default: the defaults were chosen on the tuning set, where hybrid lost and reranking cost too much latency, while the held-out set favoured lexical/hybrid/reranked retrieval. That mismatch is documented, not tuned away, and needs a new held-out set to resolve.
- The default answer generator is extractive, not an LLM. The optional LLM client is untested against a live endpoint.
- The API has no authentication, rate limiting or multi-tenancy. The Compose file uses local-only demo credentials (`policy_rag_local_only`); they are not secrets.
- The held-out set is small and written by the system's author; it is not an independent benchmark.
- Not production-ready. It is a portfolio project that demonstrates the retrieval design, the evidence gate and its evaluation.

## Configuration

Copy [`.env.example`](.env.example) to `.env` to override anything; every value in it is the application default. Docker Compose forwards these to the API container:

| Group | Variables |
|---|---|
| PostgreSQL | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST_PORT` (Compose builds `DATABASE_URL` from them) |
| Embeddings | `EMBEDDING_MODEL` (also a build argument, because the image downloads the model at build time), `EMBEDDING_DIM` |
| Chunking | `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| Retrieval | `RETRIEVAL_MODE`, `RRF_K`, `HYBRID_CANDIDATES`, `RERANK_ENABLED`, `RERANK_MODEL` (also a build argument, because the image downloads it), `RERANK_CANDIDATES`, `HNSW_M`, `HNSW_EF_CONSTRUCTION`, `HNSW_EF_SEARCH` |
| Evidence gate / RAG | `EVIDENCE_MIN_SIMILARITY`, `RAG_MAX_CONTEXT_CHUNKS` |
| Optional LLM | `LLM_PROVIDER` (`extractive` default, or `openai_compatible`), `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` |

Settings that exist in the application but are deliberately not forwarded by Compose: `DATABASE_URL` (derived from the `POSTGRES_*` values), `EMBEDDING_PROVIDER` (only `fastembed` is implemented), `SAMPLE_DATA_DIR`, `AUTO_INIT_SCHEMA`, `DB_POOL_MIN_SIZE`, `DB_POOL_MAX_SIZE` and `LLM_TIMEOUT_S`. They work when running the app directly, not through Compose.

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
tests/unit  tests/integration
```
