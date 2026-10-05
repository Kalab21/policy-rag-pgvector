"""RAG service: runs the graph and shapes its result for the API."""

import time
from collections.abc import Mapping
from dataclasses import dataclass

from app.models.domain import RetrievedChunk
from app.observability.logs import log_event
from app.observability.telemetry import current, set_attributes, span
from app.rag.evidence import EvidenceAssessment
from app.rag.generator import AnswerGenerator
from app.rag.graph import AnswerStatus, build_graph, run_graph
from app.retrieval.service import RetrievalService


@dataclass(frozen=True)
class AskResult:
    question: str
    answer: str
    status: AnswerStatus
    refusal_reason: str | None
    evidence: EvidenceAssessment
    sources: list[tuple[int, RetrievedChunk]]  # (citation number, chunk) actually cited
    retrieved_chunk_ids: list[int]
    invalid_citations: list[int]
    generator: str
    embedding_model: str


class RagService:
    def __init__(
        self,
        retrieval: RetrievalService,
        generator: AnswerGenerator,
        min_similarity: float,
        max_context_chunks: int,
    ) -> None:
        self._retrieval = retrieval
        self._generator = generator
        self._graph = build_graph(retrieval, generator, min_similarity, max_context_chunks)

    def ask(
        self, question: str, top_k: int = 5, filters: Mapping[str, str] | None = None
    ) -> AskResult:
        # Answer from current policy unless the caller explicitly asks for another status.
        effective = {"status": "current", **(filters or {})}
        telemetry = current()
        started = time.perf_counter()
        attributes = {
            "top_k": top_k,
            "filters.count": len(effective),
            "retrieval.mode": self._retrieval.mode,
            "rerank.enabled": self._retrieval.rerank_enabled,
            "generator": self._generator.name,
            **telemetry.query_attributes(question),
        }
        with span("rag.ask", attributes) as sp:
            state = run_graph(self._graph, question, top_k, effective)
            evidence = state["evidence"]
            set_attributes(
                sp,
                {
                    "answer.status": state["status"],
                    "refusal_reason": state.get("refusal_reason"),
                    "retrieved.count": len(state["retrieved"]),
                    "cited.count": len(state.get("cited", [])),
                    "best_similarity": evidence.best_similarity,
                    "threshold": evidence.threshold,
                },
            )
        telemetry.record("asks", 1, {"status": state["status"]})
        if state["status"] == "refused":
            telemetry.record("refusals", 1, {"reason": state.get("refusal_reason", "unknown")})
        log_event(
            "rag.ask.completed",
            status=state["status"],
            refusal_reason=state.get("refusal_reason"),
            retrieved=len(state["retrieved"]),
            cited=len(state.get("cited", [])),
            best_similarity=evidence.best_similarity,
            threshold=evidence.threshold,
            generator=self._generator.name,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        context = state.get("context", [])
        return AskResult(
            question=state["query"],
            answer=state["answer"],
            status=state["status"],
            refusal_reason=state.get("refusal_reason"),
            evidence=state["evidence"],
            sources=[(n, context[n - 1]) for n in state.get("cited", [])],
            retrieved_chunk_ids=[c.chunk_id for c in state["retrieved"]],
            invalid_citations=state.get("invalid_citations", []),
            generator=self._generator.name,
            embedding_model=self._retrieval.model_name,
        )
