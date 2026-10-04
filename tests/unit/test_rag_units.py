import json

import httpx
import pytest

from app.models.domain import RetrievedChunk
from app.rag.citations import validate_citations
from app.rag.evidence import assess_evidence
from app.rag.generator import ExtractiveGenerator, OpenAICompatibleGenerator


def chunk(similarity: float, text: str = "text", chunk_id: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        document="doc",
        title="Doc",
        version="1.0",
        category="fees",
        status="current",
        section="S",
        text=text,
        distance=1 - similarity,
        similarity=similarity,
        metadata={},
    )


# --- evidence gate -------------------------------------------------------------------------


def test_no_chunks_is_insufficient() -> None:
    evidence, context = assess_evidence([], 0.5, 4)
    assert evidence.status == "insufficient"
    assert evidence.reason == "no_chunks_retrieved"
    assert context == []


def test_chunks_below_the_threshold_are_insufficient() -> None:
    evidence, context = assess_evidence([chunk(0.49), chunk(0.2)], 0.5, 4)
    assert evidence.status == "insufficient"
    assert evidence.best_similarity == pytest.approx(0.49)
    assert context == []


def test_only_chunks_at_or_above_the_threshold_become_context() -> None:
    chunks = [chunk(0.8, chunk_id=1), chunk(0.5, chunk_id=2), chunk(0.3, chunk_id=3)]
    evidence, context = assess_evidence(chunks, 0.5, 4)
    assert evidence.status == "sufficient"
    assert [c.chunk_id for c in context] == [1, 2]
    assert (evidence.chunks_considered, evidence.chunks_used) == (3, 2)


def test_context_is_capped() -> None:
    _, context = assess_evidence([chunk(0.9, chunk_id=i) for i in range(6)], 0.5, 3)
    assert len(context) == 3


# --- citation validation -------------------------------------------------------------------


def test_valid_markers_are_collected_once_in_order_of_use() -> None:
    check = validate_citations("A [2]. B [1]. C [2].", 2)
    assert check.cited == [2, 1]
    assert check.invalid == []
    assert check.grounded


def test_markers_that_point_at_no_source_are_removed() -> None:
    check = validate_citations("Real [1]. Made up [7] and [0].", 2)
    assert check.cited == [1]
    assert check.invalid == [7, 0]
    assert "[7]" not in check.answer
    assert "[0]" not in check.answer


def test_an_answer_with_no_valid_citation_is_not_grounded() -> None:
    assert not validate_citations("No markers at all.", 3).grounded
    assert not validate_citations("Only a bad one [9].", 3).grounded


# --- extractive generator ------------------------------------------------------------------


def test_extractive_answer_quotes_the_matching_sentence_with_its_source_number() -> None:
    chunks = [
        chunk(0.7, "## Fees\nThe late fee is $35. Returned payments cost $25."),
        chunk(0.6, "Statements are issued monthly. Paper statements cost $3 per month."),
    ]
    answer = ExtractiveGenerator().generate("What is the late fee?", chunks)
    assert "The late fee is $35. [1]" in answer
    assert "Paper statements" not in answer
    assert "#" not in answer


def test_extractive_answer_cites_the_chunk_the_sentence_came_from() -> None:
    chunks = [chunk(0.7, "Unrelated sentence."), chunk(0.6, "Paper statements cost $3 per month.")]
    answer = ExtractiveGenerator().generate("paper statement cost", chunks)
    assert answer.endswith("[2]")


def test_extractive_with_nothing_to_quote_returns_empty() -> None:
    assert ExtractiveGenerator().generate("anything", [chunk(0.9, "## Heading only")]) == ""


# --- OpenAI-compatible generator -----------------------------------------------------------


def test_openai_compatible_generator_sends_numbered_sources_and_parses_the_reply() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " It is $35 [1]. "}}]})

    generator = OpenAICompatibleGenerator(
        "http://llm.test/v1/", "m", "k", transport=httpx.MockTransport(handler)
    )
    answer = generator.generate("late fee?", [chunk(0.9, "The late fee is $35.")])
    assert answer == "It is $35 [1]."
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == "m"
    user_message = body["messages"][1]["content"]
    assert "[1]" in user_message
    assert "The late fee is $35." in user_message
    assert generator.name == "openai-compatible:m"


def test_openai_compatible_generator_raises_on_http_errors() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    generator = OpenAICompatibleGenerator("http://llm.test/v1", "m", transport=transport)
    with pytest.raises(httpx.HTTPStatusError):
        generator.generate("q", [chunk(0.9)])
