"""Evaluate retrieval, the evidence gate and /api/ask behaviour on the gold question set.

    python -m scripts.evaluate_retrieval [--gold eval/gold.json] [--output eval/results.json]

Requires the policies to be ingested first (python -m scripts.ingest). All numbers are
measured against the live database and embedding model and printed as-is.
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
from app.retrieval.service import RetrievalService
from app.retrieval.store import supports_iterative_scan


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    ks = list(report["retrieval_current_policy"]["hit_at_k"])
    for key, title in (
        ("retrieval_all_versions", "Retrieval, no status filter (superseded policy can match)"),
        ("retrieval_current_policy", "Retrieval, status=current (what /api/ask uses)"),
    ):
        r = report[key]
        lines += [f"\n{title}: {r['questions']} answerable questions", ""]
        lines.append("| Metric | " + " | ".join(f"@{k}" for k in ks) + " |")
        lines.append("|---|" + "---|" * len(ks))
        lines.append("| Hit@K | " + " | ".join(_pct(r["hit_at_k"][k]) for k in ks) + " |")
        lines.append("| Recall@K | " + " | ".join(_pct(r["recall_at_k"][k]) for k in ks) + " |")
        lines.append(f"\nMRR: {r['mrr']:.3f}")
        if r["missed_at_max_k"]:
            lines.append(f"Not found in top {ks[-1]}: {', '.join(r['missed_at_max_k'])}")

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
        "\nEnd to end (/api/ask logic with the configured threshold and generator)",
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
    parser.add_argument("--gold", default="eval/gold.json")
    parser.add_argument("--output", help="also write the full report as JSON to this path")
    args = parser.parse_args()

    settings = get_settings()
    gold = load_gold(Path(args.gold))
    provider = get_embedding_provider(settings)
    generator = get_generator(settings, provider)
    pool = create_pool(settings.database_url, 1, 2)
    try:
        documents = list_documents(pool)
        if not documents:
            raise SystemExit("no documents found; run `python -m scripts.ingest` first")
        with pool.connection() as conn:
            version = pgvector_version(conn)
        retrieval = RetrievalService(
            pool, provider, settings.hnsw_ef_search, supports_iterative_scan(version)
        )
        rag = RagService(
            retrieval, generator, settings.evidence_min_similarity, settings.rag_max_context_chunks
        )
        report = run_evaluation(retrieval, rag, gold)
        report["setup"] = {
            "documents": len(documents),
            "chunks": sum(d["chunks"] for d in documents),
            "embedding_model": provider.model_name,
            "embedding_dim": provider.dimension,
            "pgvector_version": version,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "evidence_min_similarity": settings.evidence_min_similarity,
            "generator": generator.name,
        }
    finally:
        pool.close()

    setup = report["setup"]
    print(
        f"{setup['documents']} documents, {setup['chunks']} chunks; "
        f"{setup['embedding_model']} ({setup['embedding_dim']} dims); "
        f"pgvector {setup['pgvector_version']}; generator {setup['generator']}; "
        f"gate {setup['evidence_min_similarity']}"
    )
    print(render(report))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nfull report written to {args.output}")


if __name__ == "__main__":
    main()
