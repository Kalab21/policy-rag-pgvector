"""Run the gold set against retrieval, the evidence gate and the full ask flow.

Everything here is measured against the live database; nothing is estimated.
"""

from collections.abc import Sequence
from typing import Any

from app.evaluation.gold import AnswerableQuestion, ChunkKey, GoldSet
from app.evaluation.metrics import hit_at_k, mean, recall_at_k, reciprocal_rank
from app.models.domain import RetrievedChunk
from app.rag.service import RagService
from app.retrieval.service import RetrievalService

DEFAULT_KS = (1, 3, 5)
DEFAULT_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65)


def chunk_key(chunk: RetrievedChunk) -> ChunkKey:
    return (chunk.document, chunk.version, chunk.section)


def _filters(question: AnswerableQuestion, current_only: bool) -> dict[str, str]:
    filters = dict(question.filters)
    if current_only:
        filters.setdefault("status", "current")
    return filters


def evaluate_retrieval(
    retrieval: RetrievalService,
    questions: Sequence[AnswerableQuestion],
    ks: Sequence[int],
    current_only: bool,
) -> dict[str, Any]:
    top_k = max(ks)
    hits: dict[int, list[float]] = {k: [] for k in ks}
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    rr: list[float] = []
    misses: list[str] = []
    for q in questions:
        relevant = set(q.relevant)
        ranked = [
            chunk_key(c) for c in retrieval.search(q.question, top_k, _filters(q, current_only))
        ]
        for k in ks:
            hits[k].append(hit_at_k(ranked, relevant, k))
            recalls[k].append(recall_at_k(ranked, relevant, k))
        rr.append(reciprocal_rank(ranked, relevant))
        if hit_at_k(ranked, relevant, top_k) == 0.0:
            misses.append(q.id)
    return {
        "questions": len(questions),
        "hit_at_k": {str(k): mean(v) for k, v in hits.items()},
        "recall_at_k": {str(k): mean(v) for k, v in recalls.items()},
        "mrr": mean(rr),
        "missed_at_max_k": misses,
    }


def top1_similarity(retrieval: RetrievalService, question: str, filters: dict[str, str]) -> float:
    hits = retrieval.search(question, 1, filters)
    return hits[0].similarity if hits else 0.0


def evaluate_gate(
    retrieval: RetrievalService, gold: GoldSet, thresholds: Sequence[float]
) -> list[dict[str, Any]]:
    """For each threshold: how many answerable questions pass the gate and how many
    unanswerable ones are stopped by it. Uses the top-1 similarity under current policy."""
    answerable = [
        top1_similarity(retrieval, q.question, _filters(q, True)) for q in gold.answerable
    ]
    unanswerable = [
        top1_similarity(retrieval, q.question, {"status": "current"}) for q in gold.unanswerable
    ]
    rows: list[dict[str, Any]] = [
        {
            "threshold": t,
            "answerable_pass_rate": mean([1.0 if s >= t else 0.0 for s in answerable]),
            "unanswerable_refused_rate": mean([1.0 if s < t else 0.0 for s in unanswerable]),
        }
        for t in thresholds
    ]
    rows.append(
        {
            "similarity_range": {
                "answerable_min": min(answerable),
                "unanswerable_max": max(unanswerable),
            }
        }
    )
    return rows


def evaluate_ask(rag: RagService, gold: GoldSet, top_k: int) -> dict[str, Any]:
    correct = wrongly_refused = wrong_answer = 0
    answerable_problems: list[str] = []
    for q in gold.answerable:
        result = rag.ask(q.question, top_k, q.filters)
        if result.status == "refused":
            wrongly_refused += 1
            answerable_problems.append(f"{q.id}: wrongly refused ({result.refusal_reason})")
            continue
        text = result.answer.lower()
        has_facts = all(fact.lower() in text for fact in q.answer_contains)
        cites_relevant = any(chunk_key(c) in q.relevant for _, c in result.sources)
        if has_facts and cites_relevant:
            correct += 1
        else:
            wrong_answer += 1
            answerable_problems.append(
                f"{q.id}: answered, contains expected facts={has_facts}, "
                f"cites a relevant chunk={cites_relevant}"
            )

    refused = answered_anyway = 0
    unanswerable_problems: list[str] = []
    for u in gold.unanswerable:
        result = rag.ask(u.question, top_k, {})
        if result.status == "refused":
            refused += 1
        else:
            answered_anyway += 1
            unanswerable_problems.append(f"{u.id} ({u.kind}): answered: {u.question}")

    n_a, n_u = len(gold.answerable), len(gold.unanswerable)
    return {
        "answerable": {
            "questions": n_a,
            "answered_correctly": correct,
            "answered_incorrectly": wrong_answer,
            "wrongly_refused": wrongly_refused,
            "correct_rate": correct / n_a,
            "problems": answerable_problems,
        },
        "unanswerable": {
            "questions": n_u,
            "refused": refused,
            "answered_anyway": answered_anyway,
            "refusal_rate": refused / n_u,
            "problems": unanswerable_problems,
        },
    }


def run_evaluation(
    retrieval: RetrievalService,
    rag: RagService,
    gold: GoldSet,
    ks: Sequence[int] = DEFAULT_KS,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    return {
        "retrieval_all_versions": evaluate_retrieval(retrieval, gold.answerable, ks, False),
        "retrieval_current_policy": evaluate_retrieval(retrieval, gold.answerable, ks, True),
        "evidence_gate": evaluate_gate(retrieval, gold, thresholds),
        "ask": evaluate_ask(rag, gold, max(ks)),
    }
