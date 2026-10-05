"""Evaluate retrieval, the evidence gate and /api/ask behaviour on the tuning and held-out sets.

    python -m scripts.evaluate_retrieval [--sets tuning,held_out] [--output eval/results.json]

Requires the policies to be ingested first (python -m scripts.ingest). All numbers are
measured against the live database and models and printed as-is. See eval/README.md for how
the two sets may and may not be used.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.db.pool import create_pool
from app.db.schema import pgvector_version
from app.embeddings.factory import get_embedding_provider
from app.evaluation.gold import load_gold
from app.evaluation.runner import run_evaluation
from app.ingestion.catalog import list_documents
from app.rag.factory import get_generator
from app.rag.service import RagService
from app.retrieval.rerank import CrossEncoderReranker
from app.retrieval.service import RetrievalService
from app.retrieval.store import supports_iterative_scan

SETS = {"tuning": "eval/tuning.json", "held_out": "eval/held_out.json"}


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_set(name: str, report: dict[str, Any]) -> str:
    lines: list[str] = [f"\n==== {name} set ===="]
    ks = list(report["retrieval_current_policy"]["hit_at_k"])
    n = report["retrieval_current_policy"]["questions"]
    lines += [f"\nRetrieval by configuration, status=current ({n} answerable questions)", ""]
    header = ["Configuration"]
    header += [f"Hit@{k}" for k in ks] + [f"Recall@{k}" for k in ks] + [f"nDCG@{k}" for k in ks]
    lines.append("| " + " | ".join([*header, "MRR"]) + " |")
    lines.append("|" + "---|" * (len(header) + 1))
    for cfg, r in report["retrieval_modes"].items():
        cells = [_pct(r["hit_at_k"][k]) for k in ks]
        cells += [_pct(r["recall_at_k"][k]) for k in ks]
        cells += [f"{r['ndcg_at_k'][k]:.3f}" for k in ks]
        lines.append(f"| {cfg} | " + " | ".join(cells) + f" | {r['mrr']:.3f} |")
        if r["missed_at_max_k"]:
            lines.append(f"|  (not in top {ks[-1]}: {', '.join(r['missed_at_max_k'])}) |")

    lines += ["\nLatency per search (ms, this machine, indicative only)", ""]
    lines += ["| Configuration | mean | p50 | p95 |", "|---|---|---|---|"]
    for cfg, t in report["latency"].items():
        lines.append(f"| {cfg} | {t['mean_ms']:.0f} | {t['p50_ms']:.0f} | {t['p95_ms']:.0f} |")

    lines += ["\nEvidence gate sweep (top-1 cosine similarity, status=current)", ""]
    lines += ["| Threshold | Answerable passing | Unanswerable refused |", "|---|---|---|"]
    for row in report["evidence_gate"]:
        if "threshold" in row:
            lines.append(
                f"| {row['threshold']:.2f} | {_pct(row['answerable_pass_rate'])} "
                f"| {_pct(row['unanswerable_refused_rate'])} |"
            )
        else:
            span = row["similarity_range"]
            lines.append(
                f"\nLowest answerable top-1 similarity: {span['answerable_min']:.3f}; "
                f"highest unanswerable: {span['unanswerable_max']:.3f}"
            )

    a, u = report["ask"]["answerable"], report["ask"]["unanswerable"]
    lines += [
        "\nEnd to end (/api/ask logic with the configured threshold, mode and generator)",
        "",
        f"- Answerable: {a['answered_correctly']}/{a['questions']} answered correctly with a "
        f"relevant citation; {a['wrongly_refused']} wrongly refused; "
        f"{a['answered_incorrectly']} answered incorrectly",
        f"- Unanswerable: {u['refused']}/{u['questions']} refused; "
        f"{u['answered_anyway']} answered anyway",
    ]
    lines += [f"  - {problem}" for problem in a["problems"] + u["problems"]]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--sets", default="tuning,held_out", help="comma-separated: tuning,held_out"
    )
    parser.add_argument("--output", help="also write the full report as JSON to this path")
    args = parser.parse_args()
    wanted = [s.strip() for s in args.sets.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in SETS]
    if unknown:
        raise SystemExit(f"unknown set(s) {unknown}; choose from {sorted(SETS)}")

    settings = get_settings()
    provider = get_embedding_provider(settings)
    generator = get_generator(settings, provider)
    pool = create_pool(settings.database_url, 1, 2)
    report: dict[str, Any] = {}
    try:
        documents = list_documents(pool)
        if not documents:
            raise SystemExit("no documents found; run `python -m scripts.ingest` first")
        with pool.connection() as conn:
            version = pgvector_version(conn)
        retrieval = RetrievalService(
            pool,
            provider,
            settings.hnsw_ef_search,
            supports_iterative_scan(version),
            mode=settings.retrieval_mode,
            rrf_k=settings.rrf_k,
            candidates=settings.hybrid_candidates,
            reranker=CrossEncoderReranker(settings.rerank_model),
            rerank_enabled=settings.rerank_enabled,
            rerank_candidates=settings.rerank_candidates,
        )
        rag = RagService(
            retrieval, generator, settings.evidence_min_similarity, settings.rag_max_context_chunks
        )
        report["setup"] = {
            "documents": len(documents),
            "chunks": sum(d["chunks"] for d in documents),
            "embedding_model": provider.model_name,
            "embedding_dim": provider.dimension,
            "pgvector_version": version,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retrieval_mode": settings.retrieval_mode,
            "rerank_enabled": settings.rerank_enabled,
            "rerank_model": settings.rerank_model,
            "rerank_candidates": settings.rerank_candidates,
            "rrf_k": settings.rrf_k,
            "evidence_min_similarity": settings.evidence_min_similarity,
            "generator": generator.name,
        }
        for name in wanted:
            gold = load_gold(Path(SETS[name]))
            report[name] = run_evaluation(retrieval, rag, gold)
            report[name]["questions"] = {
                "answerable": len(gold.answerable),
                "unanswerable": len(gold.unanswerable),
            }
    finally:
        pool.close()

    s = report["setup"]
    print(
        f"{s['documents']} documents, {s['chunks']} chunks; {s['embedding_model']} "
        f"({s['embedding_dim']} dims); pgvector {s['pgvector_version']}; default mode "
        f"{s['retrieval_mode']}, rerank {'on' if s['rerank_enabled'] else 'off'}; "
        f"generator {s['generator']}; gate {s['evidence_min_similarity']}"
    )
    for name in wanted:
        print(render_set(name, report[name]))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nfull report written to {args.output}")


if __name__ == "__main__":
    main()
