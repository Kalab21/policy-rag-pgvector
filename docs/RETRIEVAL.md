# Retrieval and grounding

How documents are stored, chunked, embedded and retrieved, and how the RAG flow decides whether to answer. Security filtering applied inside these queries is described in [SECURITY.md](SECURITY.md); settings are listed in [OPERATIONS.md](OPERATIONS.md#configuration); measured quality is in [`eval/README.md`](../eval/README.md).

## Components

| Area | Implementation |
|---|---|
| Vector store | PostgreSQL 16 + pgvector (`pgvector/pgvector:pg16`, 0.8.7 in the measured runs), `vector(384)` column |
| Index | HNSW, `vector_cosine_ops`, `m=16`, `ef_construction=64`; `hnsw.ef_search=40` per query |
| Similarity | Cosine distance (`<=>`); similarity = 1 − distance |
| Retrieval modes | `semantic` (pgvector, default), `lexical` (PostgreSQL full-text), `hybrid` (both, fused with Reciprocal Rank Fusion), selectable per request |
| Reranking | Optional local cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2` via fastembed, ONNX) re-orders a candidate set from any retrieval mode; off by default |
| Metadata filtering | In the same SQL statement: `metadata @> filter` (JSONB, GIN-indexed) on `category`, `version`, `status`, `document`, `section` |
| Filtered ANN | pgvector ≥ 0.8 iterative scan (`hnsw.iterative_scan = strict_order`) lets HNSW keep scanning for matching rows after the filter, which helps fill `top_k` when at least `top_k` matching rows exist |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via [fastembed](https://github.com/qdrant/fastembed) (ONNX, runs locally on CPU), behind an `EmbeddingProvider` interface. Model name, dimension and a content hash are stored per document |
| Ingestion | Markdown with front matter → normalise → section-aware chunking → embed → store. Idempotent (see below) |
| RAG | LangGraph state machine: `validate_query → retrieve → assess_evidence → generate_answer \| refuse → validate_citations` |

Only LangGraph is used from the LangChain ecosystem; there is no other LangChain code in the app.

## Schema

`documents` (name, title, version, category, status, `content_hash`, `embedding_model`, `embedding_dim`) and `chunks` (`chunk_text`, `chunk_hash`, JSONB `metadata`, `embedding vector(384)`). The API refuses to start if the column width and `EMBEDDING_DIM` disagree. On the 34-chunk sample PostgreSQL may choose an exact scan; use of the HNSW index is verified with `enable_seqscan=off`.

## Chunking

Split on markdown headings, then pack whole sentences up to 700 characters with a 120-character overlap; over-long sentences split on words. The text embedded is `"{title} | {section}\n{chunk}"`, while the stored and cited text is the plain chunk. On the bundled documents each section fits in one chunk, so the overlap is never exercised.

## Idempotent ingestion

A document's fingerprint covers its content, the embedding model and the chunk settings. Running ingest again skips unchanged documents without re-embedding, replaces a changed document atomically, and leaves existing rows untouched if embedding fails.

## Retrieval modes

`semantic` embeds the query and searches pgvector. `lexical` uses a PostgreSQL full-text index (a generated `tsvector` column with a GIN index, English stemming, stop words dropped, any content word may match). `hybrid` runs both under the same metadata filter and merges the two ranked lists with Reciprocal Rank Fusion (`score = Σ 1/(60 + rank)`), which uses ranks only, so cosine similarities and text-search ranks never have to be compared.

Why bother: dense embeddings capture meaning but can rank a chunk containing an exact identifier or rare term below vaguely related text; an integration test builds that situation with hand-made vectors, and hybrid retrieval surfaces the chunk.

Choose a mode with `RETRIEVAL_MODE` or `"mode"` in a `/api/search` request. Results from every mode carry a real cosine similarity, because the evidence gate depends on it.

```bash
curl -s localhost:8000/api/search -H 'content-type: application/json' -d '{
  "query": "What is the late payment fee?",
  "top_k": 3,
  "filters": {"category": "fees"}
}'
```

Each result has the chunk `text`, its source (`document`, `title`, `version`, `section`), `category`/`status`, `distance`, and `similarity`.

## Reranking

The retrievers are fast because they score the query and each chunk independently. A cross-encoder reads the query and one chunk together, which is more accurate but too slow for the whole corpus, so it only re-orders a candidate set: the retriever returns `RERANK_CANDIDATES` chunks (never fewer than `top_k`), the cross-encoder scores them, and the best `top_k` are kept. Metadata filters are applied during candidate retrieval, so a filtered-out chunk is never scored. Reranked results keep their cosine similarity, so the evidence gate behaves as before. Enable it with `RERANK_ENABLED=true` or `"rerank": true` on a `/api/search` request.

## HNSW and pgvector tuning

- `HNSW_M` and `HNSW_EF_CONSTRUCTION` only apply when the index is created; `HNSW_EF_SEARCH` applies to every query.
- Filtered queries use pgvector's iterative scan (`hnsw.iterative_scan = strict_order`, pgvector ≥ 0.8) so the index keeps scanning past rows removed by the metadata filter.
- Changing `EMBEDDING_MODEL` to one with a different vector width needs `EMBEDDING_DIM` changed and the `chunks` table recreated (see [OPERATIONS.md](OPERATIONS.md#configuration-notes)).

## Retrieval is separate from RAG

`app/retrieval` knows nothing about LLMs or the graph; `app/rag` calls it as a service.

## Evidence gate

Before any text is generated, the best chunk's cosine similarity must reach `EVIDENCE_MIN_SIMILARITY` (default `0.55`). Chunks below it are never given to the generator. The evidence gate is evaluated separately from retrieval and is covered by regression tests across answerable and unanswerable questions.

## Citation check

The generator cites sources as `[1]`, `[2]`. Markers that do not match a supplied chunk are removed, and an answer left with no valid citation is refused. Every generator (see [OPERATIONS.md](OPERATIONS.md#generator-providers)) only ever receives chunks that passed the evidence gate, and every answer goes through the same citation check afterwards.

## Ask: grounded answer or refusal

```bash
curl -s localhost:8000/api/ask -H 'content-type: application/json' \
  -d '{"question": "What is the late payment fee?"}'
```

Returns `answer`, `status` (`answered` or `refused`), `sources` (the cited chunks), `retrieved_chunk_ids`, and `evidence` (status, reason, best similarity, threshold). A question the documents do not cover, such as *"How do I bake sourdough bread?"*, comes back `refused` with no sources. Answers use **current** policy unless `filters.status` says otherwise (`"superseded"` returns the older underwriting rules).
