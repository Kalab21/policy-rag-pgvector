import pytest

from app.retrieval.rerank import CrossEncoderReranker, passage_text, rerank_chunks
from tests.fakes import KeywordReranker
from tests.unit.test_rag_units import chunk


def make(chunk_id: int, text: str, similarity: float = 0.5):  # type: ignore[no-untyped-def]
    return chunk(similarity, text, chunk_id)


def test_chunks_are_reordered_by_reranker_score() -> None:
    chunks = [make(1, "unrelated words"), make(2, "late fee"), make(3, "the late payment fee")]
    out = rerank_chunks(KeywordReranker(), "late payment fee", chunks, top_k=3)
    assert [c.chunk_id for c in out] == [3, 2, 1]
    assert [c.rerank_score for c in out] == [3.0, 2.0, 0.0]


def test_only_top_k_are_kept() -> None:
    chunks = [make(i, "late fee" if i == 4 else "nothing") for i in range(1, 6)]
    out = rerank_chunks(KeywordReranker(), "late fee", chunks, top_k=2)
    assert len(out) == 2
    assert out[0].chunk_id == 4


def test_equal_scores_are_ordered_by_chunk_id_so_results_are_deterministic() -> None:
    chunks = [make(9, "same"), make(2, "same"), make(5, "same")]
    first = rerank_chunks(KeywordReranker(), "same", chunks, top_k=3)
    second = rerank_chunks(KeywordReranker(), "same", list(reversed(chunks)), top_k=3)
    assert [c.chunk_id for c in first] == [2, 5, 9] == [c.chunk_id for c in second]


def test_reranking_keeps_the_cosine_similarity_the_evidence_gate_uses() -> None:
    out = rerank_chunks(KeywordReranker(), "x", [make(1, "x", similarity=0.73)], top_k=1)
    assert out[0].similarity == pytest.approx(0.73)


def test_an_empty_candidate_set_does_not_call_the_model() -> None:
    reranker = KeywordReranker()
    assert rerank_chunks(reranker, "anything", [], top_k=3) == []
    assert reranker.calls == []


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="top_k"):
        rerank_chunks(KeywordReranker(), "x", [make(1, "x")], top_k=0)


def test_the_cross_encoder_reads_title_section_and_text() -> None:
    assert passage_text(make(1, "body text")) == "Doc | S\nbody text"


class FakeOnnxModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.seen: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.seen.append((query, documents))
        return self.scores


def test_the_adapter_passes_query_and_passages_to_the_model_and_returns_floats() -> None:
    model = FakeOnnxModel([1.5, -2.0])
    reranker = CrossEncoderReranker("m", model=model)
    assert reranker.score("q", ["a", "b"]) == [1.5, -2.0]
    assert model.seen == [("q", ["a", "b"])]
    assert reranker.model_name == "m"


def test_the_adapter_rejects_a_model_that_returns_the_wrong_number_of_scores() -> None:
    reranker = CrossEncoderReranker("m", model=FakeOnnxModel([1.0]))
    with pytest.raises(RuntimeError, match="wrong number"):
        reranker.score("q", ["a", "b"])


def test_the_adapter_does_not_load_a_model_until_it_is_used() -> None:
    reranker = CrossEncoderReranker("a-model-that-does-not-exist")  # must not raise or download
    assert reranker.score("q", []) == []  # empty input never needs the model
