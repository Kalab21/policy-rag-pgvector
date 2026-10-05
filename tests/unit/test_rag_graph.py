"""The LangGraph flow, with a scripted retrieval service so each branch can be forced.

Real pgvector retrieval is covered by the integration tests; here only routing is tested.
"""

from typing import Any

import pytest

from app.models.domain import RetrievedChunk
from app.rag.graph import NOT_ENOUGH_MARKER, REFUSAL_MESSAGE, build_graph, run_graph
from tests.unit.test_rag_units import chunk


class ScriptedRetrieval:
    model_name = "scripted"

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int, dict[str, str]]] = []

    def search(
        self, query: str, top_k: int, filters: dict[str, str], **_kwargs: object
    ) -> list[RetrievedChunk]:
        self.calls.append((query, top_k, filters))
        return self.chunks


class ScriptedGenerator:
    name = "scripted"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        self.calls += 1
        return self.answer


def run(
    chunks: list[RetrievedChunk], answer: str, question: str = " q? "
) -> tuple[dict[str, Any], ScriptedRetrieval, ScriptedGenerator]:
    retrieval, generator = ScriptedRetrieval(chunks), ScriptedGenerator(answer)
    graph = build_graph(retrieval, generator, 0.5, 4)  # type: ignore[arg-type]
    state = run_graph(graph, question, 5, {"status": "current"})
    return dict(state), retrieval, generator


def test_strong_evidence_flows_through_generation_to_a_cited_answer() -> None:
    state, retrieval, generator = run([chunk(0.9, chunk_id=11)], "Yes [1].")
    assert state["status"] == "answered"
    assert state["cited"] == [1]
    assert state["query"] == "q?"  # validate_query stripped it
    assert retrieval.calls == [("q?", 5, {"status": "current"})]
    assert generator.calls == 1


def test_weak_evidence_refuses_without_calling_the_generator() -> None:
    state, _, generator = run([chunk(0.2)], "Should never be used [1].")
    assert state["status"] == "refused"
    assert state["answer"] == REFUSAL_MESSAGE
    assert state["refusal_reason"] == "insufficient_evidence"
    assert generator.calls == 0


def test_nothing_retrieved_refuses() -> None:
    state, _, _ = run([], "x [1]")
    assert state["status"] == "refused"
    assert state["evidence"].reason == "no_chunks_retrieved"


def test_an_answer_citing_a_nonexistent_source_is_refused() -> None:
    state, _, _ = run([chunk(0.9)], "Claim [4].")
    assert state["status"] == "refused"
    assert state["refusal_reason"] == "answer_not_grounded_in_sources"
    assert state["invalid_citations"] == [4]


def test_an_answer_with_no_citation_is_refused() -> None:
    state, _, _ = run([chunk(0.9)], "Confident but uncited.")
    assert state["status"] == "refused"


def test_the_generator_can_decline_and_the_graph_honours_it() -> None:
    state, _, _ = run([chunk(0.9)], NOT_ENOUGH_MARKER)
    assert state["status"] == "refused"
    assert state["refusal_reason"] == "generator_found_no_answer"


def test_invalid_markers_are_stripped_but_a_valid_one_keeps_the_answer() -> None:
    state, _, _ = run([chunk(0.9)], "Good [1]. Bad [3].")
    assert state["status"] == "answered"
    assert "[3]" not in state["answer"]


def test_a_blank_question_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        run([chunk(0.9)], "x [1]", question="   ")
