import pytest

from app.retrieval.fusion import rrf_fuse
from tests.unit.test_rag_units import chunk


def ids(results: list) -> list[int]:  # type: ignore[type-arg]
    return [c.chunk_id for c in results]


def test_a_chunk_ranked_well_by_both_retrievers_beats_one_ranked_first_by_only_one() -> None:
    semantic = [chunk(0.9, chunk_id=1), chunk(0.8, chunk_id=2), chunk(0.7, chunk_id=3)]
    lexical = [chunk(0.5, chunk_id=2), chunk(0.4, chunk_id=4)]
    fused = rrf_fuse({"semantic": semantic, "lexical": lexical})
    assert ids(fused) == [2, 1, 4, 3]


def test_scores_follow_the_rrf_formula() -> None:
    fused = rrf_fuse(
        {"a": [chunk(0.9, chunk_id=1)], "b": [chunk(0.9, chunk_id=2), chunk(0.9, chunk_id=1)]}, k=10
    )
    by_id = {c.chunk_id: c for c in fused}
    assert by_id[1].score == pytest.approx(1 / 11 + 1 / 12)
    assert by_id[2].score == pytest.approx(1 / 11)


def test_each_result_records_which_retrievers_found_it() -> None:
    fused = rrf_fuse(
        {
            "semantic": [chunk(0.9, chunk_id=1)],
            "lexical": [chunk(0.9, chunk_id=1), chunk(0.5, chunk_id=2)],
        }
    )
    by_id = {c.chunk_id: c for c in fused}
    assert by_id[1].matched_by == ("semantic", "lexical")
    assert by_id[2].matched_by == ("lexical",)


def test_ties_break_by_chunk_id_so_the_order_is_deterministic() -> None:
    # Chunks 7 and 3 each rank first in one list: identical fused scores.
    fused = rrf_fuse({"a": [chunk(0.9, chunk_id=7)], "b": [chunk(0.9, chunk_id=3)]})
    assert ids(fused) == [3, 7]
    again = rrf_fuse({"b": [chunk(0.9, chunk_id=3)], "a": [chunk(0.9, chunk_id=7)]})
    assert ids(again) == [3, 7]


def test_only_ranks_matter_not_the_scores_of_each_retriever() -> None:
    low = [chunk(0.01, chunk_id=1), chunk(0.001, chunk_id=2)]
    high = [chunk(0.99, chunk_id=1), chunk(0.98, chunk_id=2)]
    assert ids(rrf_fuse({"x": low})) == ids(rrf_fuse({"x": high})) == [1, 2]


def test_a_chunk_listed_twice_by_one_retriever_counts_once() -> None:
    fused = rrf_fuse({"a": [chunk(0.9, chunk_id=1), chunk(0.8, chunk_id=1)]}, k=10)
    assert fused[0].score == pytest.approx(1 / 11)


def test_empty_inputs_and_bad_k() -> None:
    assert rrf_fuse({}) == []
    assert rrf_fuse({"a": []}) == []
    with pytest.raises(ValueError, match="k must"):
        rrf_fuse({"a": [chunk(0.9)]}, k=0)
