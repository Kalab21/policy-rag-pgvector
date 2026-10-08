# Evaluation protocol

Two question sets, kept apart on purpose.

| File | Role |
|---|---|
| `tuning.json` | Used while building the system. Every threshold, ranking rule and default was chosen while looking at it. Numbers on this set are optimistic. |
| `held_out.json` | Written after the retrieval pipeline was fixed. **Not used to choose anything**: not the chunk size or overlap, not the evidence threshold, not the RRF constant, not the candidate depth, not the reranker settings, not the retrieval mode. It is only reported. |

Each file has `answerable` questions (with the document, version and section that answers each one, and a fact the answer must contain) and `unanswerable` questions (the system should refuse).

## Rules (written before the held-out numbers existed)

1. **Settings are chosen on `tuning.json` only.** Parameters, thresholds and which technique is the default are decided from tuning-set results plus the measured latency and the added complexity.
2. **Default rule for an optional technique (hybrid search, reranking).** It becomes the default only if, on the tuning set, it is at least as good as the current default on both MRR and Hit@1 and does not lower Hit@5, *and* its added latency per query is small enough to be worth it on this machine (under 250 ms median).
3. **The held-out set may only veto.** After the settings are fixed, the held-out set is run once. If a technique chosen as the default is worse than the previous default on held-out MRR, it is switched back to optional and that is stated. The held-out set is never used to search for better settings.
4. The held-out numbers are reported whatever they are, including when they are worse than the tuning numbers.

## What "held-out" does and does not mean here

- The questions were written by the same person who built the system, from the same seven synthetic documents. It is a held-out set, not an independent one.
- It is small. Differences of a few points are within noise.
- The corpus is 34 chunks. High scores here do not predict performance on a large corpus.

## Metrics

Relevance is binary per chunk: a retrieved chunk is relevant if its (document, version, section) is listed for the question. `Hit@K`: a relevant chunk is in the top K. `Recall@K`: the share of the relevant chunks in the top K. `MRR`: mean of 1 / rank of the first relevant chunk. `nDCG@K`: discounted cumulative gain with binary gains, normalised by the best possible ordering.

The harness reports Hit@K, Recall@K, nDCG@K, MRR, per-configuration latency, a gate sweep and end-to-end behaviour, over the tuning (33 answerable + 13 unanswerable) and held-out (45 answerable + 16 unanswerable) sets; CI floors apply to both.

Run both sets: `python -m scripts.evaluate_retrieval`.

## Latest results

Measured over the synthetic policy corpus (7 documents, 34 chunks, sentence-transformers/all-MiniLM-L6-v2, pgvector 0.8.7); CI enforces regression floors for both sets. The raw numbers are in [`results.json`](results.json).

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

Semantic retrieval is the default; lexical, hybrid and reranked configurations are selectable per request or through configuration. Reranking runs a local cross-encoder over a candidate set.

### Latency

Per-search latency from the harness (held-out run, milliseconds, one development machine):

| Configuration | p50 | p95 |
|---|---|---|
| semantic | 7 | 9 |
| lexical | 6 | 8 |
| hybrid | 10 | 12 |
| semantic+rerank | 332 | 468 |
| hybrid+rerank | 328 | 384 |

The evidence gate is evaluated separately from retrieval and is covered by regression tests across answerable and unanswerable questions.
