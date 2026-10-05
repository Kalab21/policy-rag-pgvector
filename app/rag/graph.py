"""The RAG flow as a small LangGraph state machine.

    START -> validate_query -> retrieve -> assess_evidence
                                              |-- sufficient --> generate_answer --|
                                              |-- insufficient -> refuse ---------|
                                                                                   v
                                                         validate_citations -> END

Retrieval (app.retrieval) knows nothing about this graph; the graph only calls it.
"""

import time
from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.models.domain import RetrievedChunk
from app.observability.telemetry import current, set_attributes, span
from app.rag.citations import validate_citations
from app.rag.evidence import EvidenceAssessment, assess_evidence
from app.rag.generator import (
    NOT_ENOUGH_MARKER,
    AnswerGenerator,
    GenerationError,
    GeneratorOutputError,
)
from app.retrieval.service import RetrievalService

REFUSAL_MESSAGE = (
    "I can't answer that from the policy documents available to me. "
    "The retrieved passages do not contain enough relevant evidence."
)
MAX_QUESTION_CHARS = 1000

AnswerStatus = Literal["answered", "refused"]


class RagState(TypedDict, total=False):
    question: str
    top_k: int
    filters: dict[str, str]
    query: str
    retrieved: list[RetrievedChunk]  # everything the vector search returned
    context: list[RetrievedChunk]  # the subset that passed the evidence gate
    evidence: EvidenceAssessment
    answer: str
    status: AnswerStatus
    cited: list[int]  # 1-based positions in `context`
    invalid_citations: list[int]
    refusal_reason: str
    generation_failed: bool


def build_graph(
    retrieval: RetrievalService,
    generator: AnswerGenerator,
    min_similarity: float,
    max_context_chunks: int,
) -> Any:
    def validate_query(state: RagState) -> RagState:
        query = state["question"].strip()
        if not query:
            raise ValueError("question must not be empty")
        if len(query) > MAX_QUESTION_CHARS:
            raise ValueError(f"question must be at most {MAX_QUESTION_CHARS} characters")
        return {"query": query}

    def retrieve(state: RagState) -> RagState:
        chunks = retrieval.search(state["query"], state["top_k"], state.get("filters") or {})
        return {"retrieved": chunks}

    def assess(state: RagState) -> RagState:
        with span("evidence.assess", {"threshold": min_similarity}) as sp:
            evidence, context = assess_evidence(
                state["retrieved"], min_similarity, max_context_chunks
            )
            set_attributes(
                sp,
                {
                    "evidence.status": evidence.status,
                    "best_similarity": evidence.best_similarity,
                    "chunks.considered": evidence.chunks_considered,
                    "chunks.used": evidence.chunks_used,
                },
            )
        return {"evidence": evidence, "context": context}

    def route(state: RagState) -> Literal["generate_answer", "refuse"]:
        return "generate_answer" if state["evidence"].status == "sufficient" else "refuse"

    def generate_answer(state: RagState) -> RagState:
        telemetry = current()
        provider = generator.name.split(":")[0]
        started = time.perf_counter()
        try:
            with span(
                "generator.generate",
                {
                    "generator.name": generator.name,
                    "provider": provider,
                    "chunks": len(state["context"]),
                },
            ) as sp:
                try:
                    answer = generator.generate(state["query"], state["context"])
                except GeneratorOutputError:
                    # The model answered but its output failed validation: fail closed.
                    set_attributes(sp, {"output.valid": False})
                    telemetry.record(
                        "provider_errors", 1, {"provider": provider, "error_type": "invalid_output"}
                    )
                    return {"answer": "", "status": "answered", "generation_failed": True}
                except GenerationError as exc:
                    telemetry.record(
                        "provider_errors",
                        1,
                        {"provider": provider, "error_type": type(exc).__name__},
                    )
                    raise
                set_attributes(sp, {"output.valid": True})
        finally:
            telemetry.record(
                "generation_ms", (time.perf_counter() - started) * 1000, {"provider": provider}
            )
        return {"answer": answer, "status": "answered"}

    def refuse(state: RagState) -> RagState:
        return {
            "answer": REFUSAL_MESSAGE,
            "status": "refused",
            "cited": [],
            "refusal_reason": "insufficient_evidence",
        }

    def check_citations(state: RagState) -> RagState:
        with span("citation.validate") as sp:
            result = _check_citations(state)
            outcome = result.get("status", state.get("status"))
            set_attributes(
                sp,
                {
                    "outcome": outcome,
                    "cited": len(result.get("cited", state.get("cited", []))),
                    "invalid": len(result.get("invalid_citations", [])),
                    "refusal_reason": result.get("refusal_reason", state.get("refusal_reason")),
                },
            )
            return result

    def _check_citations(state: RagState) -> RagState:
        if state["status"] == "refused":
            return {}
        answer = state.get("answer", "")
        if not answer or NOT_ENOUGH_MARKER in answer:
            return {
                "answer": REFUSAL_MESSAGE,
                "status": "refused",
                "cited": [],
                "refusal_reason": (
                    "generator_output_invalid"
                    if state.get("generation_failed")
                    else "generator_found_no_answer"
                ),
            }
        check = validate_citations(answer, len(state["context"]))
        if not check.grounded:
            return {
                "answer": REFUSAL_MESSAGE,
                "status": "refused",
                "cited": [],
                "invalid_citations": check.invalid,
                "refusal_reason": "answer_not_grounded_in_sources",
            }
        return {"answer": check.answer, "cited": check.cited, "invalid_citations": check.invalid}

    graph = StateGraph(RagState)
    graph.add_node("validate_query", validate_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("assess_evidence", assess)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("refuse", refuse)
    graph.add_node("validate_citations", check_citations)
    graph.add_edge(START, "validate_query")
    graph.add_edge("validate_query", "retrieve")
    graph.add_edge("retrieve", "assess_evidence")
    graph.add_conditional_edges("assess_evidence", route)
    graph.add_edge("generate_answer", "validate_citations")
    graph.add_edge("refuse", "validate_citations")
    graph.add_edge("validate_citations", END)
    return graph.compile()


def run_graph(compiled: Any, question: str, top_k: int, filters: Mapping[str, str]) -> RagState:
    result: RagState = compiled.invoke(
        {"question": question, "top_k": top_k, "filters": dict(filters)}
    )
    return result
