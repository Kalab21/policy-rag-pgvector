"""The RAG flow as a small LangGraph state machine.

    START -> validate_query -> retrieve -> assess_evidence
                                              |-- sufficient --> generate_answer --|
                                              |-- insufficient -> refuse ---------|
                                                                                   v
                                                         validate_citations -> END

Retrieval (app.retrieval) knows nothing about this graph; the graph only calls it.
"""

from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.models.domain import RetrievedChunk
from app.rag.citations import validate_citations
from app.rag.evidence import EvidenceAssessment, assess_evidence
from app.rag.generator import AnswerGenerator
from app.retrieval.service import RetrievalService

REFUSAL_MESSAGE = (
    "I can't answer that from the policy documents available to me. "
    "The retrieved passages do not contain enough relevant evidence."
)
NOT_ENOUGH_MARKER = "INSUFFICIENT_EVIDENCE"
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
        evidence, context = assess_evidence(state["retrieved"], min_similarity, max_context_chunks)
        return {"evidence": evidence, "context": context}

    def route(state: RagState) -> Literal["generate_answer", "refuse"]:
        return "generate_answer" if state["evidence"].status == "sufficient" else "refuse"

    def generate_answer(state: RagState) -> RagState:
        answer = generator.generate(state["query"], state["context"])
        return {"answer": answer, "status": "answered"}

    def refuse(state: RagState) -> RagState:
        return {
            "answer": REFUSAL_MESSAGE,
            "status": "refused",
            "cited": [],
            "refusal_reason": "insufficient_evidence",
        }

    def check_citations(state: RagState) -> RagState:
        if state["status"] == "refused":
            return {}
        answer = state.get("answer", "")
        if not answer or NOT_ENOUGH_MARKER in answer:
            return {
                "answer": REFUSAL_MESSAGE,
                "status": "refused",
                "cited": [],
                "refusal_reason": "generator_found_no_answer",
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
