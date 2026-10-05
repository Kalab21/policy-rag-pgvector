"""Reranking inside the retrieval service, on real PostgreSQL + pgvector.

Only the cross-encoder is replaced (a deterministic keyword scorer); search, filters and
candidate selection are the real thing.
"""

import pytest

from app.db.pool import DictPool, create_pool
from app.retrieval.service import RetrievalService
from tests.conftest import TEST_DIM
from tests.fakes import HashingEmbedder, KeywordReranker
from tests.helpers import Spec, seed_chunks

pytestmark = pytest.mark.integration

QUERY = "wire transfer limit increase requires manager approval"

TEXTS = [
    ("wire transfer limits are reviewed", "payments"),
    ("wire transfer callback verification applies", "payments"),
    ("wire transfers need approval", "payments"),
    ("wire fees apply to outgoing wire transfers", "fees"),
    ("late fees apply after the grace period", "fees"),
    ("call recordings are kept two years", "privacy"),
    ("breach notification within seventy two hours", "privacy"),
    ("complaints are acknowledged in two days", "compliance"),
    ("identity documents must be unexpired", "compliance"),
    ("wire transfer limit increase requires manager approval over ten thousand", "payments"),
]


@pytest.fixture
def pool(clean_db: str):  # type: ignore[no-untyped-def]
    pool = create_pool(clean_db)
    yield pool
    pool.close()


@pytest.fixture
def corpus(clean_db: str) -> str:
    embedder = HashingEmbedder(TEST_DIM)
    seed_chunks(
        clean_db,
        [
            Spec(t, embedder.embed_documents([t])[0], category=c, document=f"d{i}", section=f"s{i}")
            for i, (t, c) in enumerate(TEXTS)
        ],
    )
    return clean_db


def service(pool: DictPool, reranker: KeywordReranker | None, **kw) -> RetrievalService:  # type: ignore[no-untyped-def]
    return RetrievalService(pool, HashingEmbedder(TEST_DIM), reranker=reranker, **kw)


def test_reranking_reorders_and_scores_the_candidates(pool: DictPool, corpus: str) -> None:
    plain = service(pool, KeywordReranker()).search(QUERY, top_k=3)
    reranked = service(pool, KeywordReranker(), rerank_enabled=True, rerank_candidates=10).search(
        QUERY, top_k=3
    )
    assert "manager approval" in reranked[0].text
    assert reranked[0].rerank_score is not None
    assert all(c.rerank_score is None for c in plain)
    assert len(reranked) == 3


def test_the_reranker_only_sees_the_candidate_depth_not_the_whole_corpus(
    pool: DictPool, corpus: str
) -> None:
    reranker = KeywordReranker()
    service(pool, reranker, rerank_enabled=True, rerank_candidates=4).search(QUERY, top_k=2)
    assert len(reranker.calls) == 1
    assert len(reranker.calls[0][1]) == 4  # 4 of the 10 chunks


def test_candidate_depth_is_never_smaller_than_top_k(pool: DictPool, corpus: str) -> None:
    reranker = KeywordReranker()
    out = service(pool, reranker, rerank_enabled=True, rerank_candidates=2).search(QUERY, top_k=5)
    assert len(reranker.calls[0][1]) == 5
    assert len(out) == 5


def test_metadata_filters_are_applied_before_reranking(pool: DictPool, corpus: str) -> None:
    reranker = KeywordReranker()
    out = service(pool, reranker, rerank_enabled=True, rerank_candidates=10).search(
        QUERY, top_k=3, filters={"category": "fees"}
    )
    seen = [p for _, passages in reranker.calls for p in passages]
    assert seen  # fees chunks were scored
    assert not any("manager approval" in p for p in seen)  # the payments chunk never was
    assert {c.category for c in out} == {"fees"}


def test_reranking_is_off_unless_enabled(pool: DictPool, corpus: str) -> None:
    reranker = KeywordReranker()
    out = service(pool, reranker).search(QUERY, top_k=3)
    assert reranker.calls == []
    assert all(c.rerank_score is None for c in out)


def test_a_request_can_turn_reranking_on_or_off(pool: DictPool, corpus: str) -> None:
    on_by_default = KeywordReranker()
    service(pool, on_by_default, rerank_enabled=True).search(QUERY, 3, rerank=False)
    assert on_by_default.calls == []
    off_by_default = KeywordReranker()
    service(pool, off_by_default).search(QUERY, 3, rerank=True)
    assert len(off_by_default.calls) == 1


@pytest.mark.parametrize("mode", ["semantic", "lexical", "hybrid"])
def test_every_retrieval_mode_can_feed_the_reranker(pool: DictPool, corpus: str, mode: str) -> None:
    reranker = KeywordReranker()
    svc = service(pool, reranker, rerank_enabled=True, rerank_candidates=6)
    out = svc.search(QUERY, top_k=2, mode=mode)  # type: ignore[arg-type]
    assert len(out) == 2
    assert all(c.rerank_score is not None for c in out)


def test_an_empty_retrieval_returns_nothing_without_calling_the_model(
    pool: DictPool, corpus: str
) -> None:
    reranker = KeywordReranker()
    out = service(pool, reranker, rerank_enabled=True).search(
        QUERY, top_k=3, filters={"category": "does-not-exist"}
    )
    assert out == []
    assert reranker.calls == []


def test_requesting_reranking_without_a_reranker_is_an_error(pool: DictPool, corpus: str) -> None:
    with pytest.raises(ValueError, match="no reranker"):
        service(pool, None).search(QUERY, 3, rerank=True)


def test_reranked_results_are_deterministic(pool: DictPool, corpus: str) -> None:
    svc = service(pool, KeywordReranker(), rerank_enabled=True, rerank_candidates=10)
    runs = [[c.chunk_id for c in svc.search(QUERY, top_k=5)] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
