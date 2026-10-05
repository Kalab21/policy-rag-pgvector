"""Lexical (full-text) and hybrid retrieval against real PostgreSQL + pgvector.

Vectors are hand-made so the semantic ranking is known exactly, and the texts are written so
the lexical ranking differs from it. That is what makes fusion observable.
"""

import psycopg
import pytest

from app.db.pool import DictPool, create_pool
from app.db.schema import init_schema
from app.retrieval.store import search_chunks, search_hybrid, search_lexical
from tests.conftest import TEST_DIM
from tests.helpers import Spec, basis, seed_chunks

pytestmark = pytest.mark.integration

QUERY_VECTOR = basis(0)
IDENTIFIER_QUERY = "What does ADR-0002 require?"


@pytest.fixture
def pool(clean_db: str):  # type: ignore[no-untyped-def]
    pool = create_pool(clean_db)
    yield pool
    pool.close()


@pytest.fixture
def corpus(clean_db: str) -> str:
    """The chunk holding the exact identifier is semantically FAR from the query vector."""
    seed_chunks(
        clean_db,
        [
            Spec(
                "Wire transfers above the limit need callback verification under ADR-0002.",
                basis(5),
                category="payments",
                section="Wires",
            ),
            Spec("General guidance about payments and transfers.", basis(0), category="payments"),
            Spec(
                "Late fees are charged after the grace period ends.",
                basis(0, tilt=0.2),
                category="fees",
                section="Late fees",
            ),
            Spec("Call recordings are retained for two years.", basis(1), category="privacy"),
            Spec(
                "Wire transfer limits are reviewed every year.",
                basis(0, tilt=0.5),
                category="fees",
                section="Limits",
            ),
        ],
    )
    return clean_db


def texts(results: list) -> list[str]:  # type: ignore[type-arg]
    return [c.text for c in results]


def test_dense_retrieval_alone_misses_the_exact_identifier(pool: DictPool, corpus: str) -> None:
    hits = search_chunks(pool, QUERY_VECTOR, top_k=2)
    assert not any("ADR-0002" in c.text for c in hits)


def test_lexical_search_finds_the_exact_identifier(pool: DictPool, corpus: str) -> None:
    hits = search_lexical(pool, IDENTIFIER_QUERY, QUERY_VECTOR, top_k=3)
    assert "ADR-0002" in hits[0].text
    assert hits[0].matched_by == ("lexical",)
    assert hits[0].score is not None
    assert hits[0].score > 0


def test_lexical_results_still_report_a_cosine_similarity(pool: DictPool, corpus: str) -> None:
    hit = search_lexical(pool, IDENTIFIER_QUERY, QUERY_VECTOR, top_k=1)[0]
    assert hit.similarity == pytest.approx(1.0 - hit.distance)
    assert hit.distance == pytest.approx(1.0, abs=1e-6)  # basis(5) is orthogonal to basis(0)


def test_lexical_matching_uses_stemming_and_ignores_stop_words(pool: DictPool, corpus: str) -> None:
    hits = search_lexical(pool, "How long are the recordings of calls retained?", QUERY_VECTOR, 3)
    assert "Call recordings" in hits[0].text


def test_lexical_matches_any_content_word_not_all_of_them(pool: DictPool, corpus: str) -> None:
    # "grace" is in one chunk; "unicorns" is in none. An AND query would return nothing.
    hits = search_lexical(pool, "grace unicorns", QUERY_VECTOR, top_k=3)
    assert texts(hits) == ["Late fees are charged after the grace period ends."]


def test_a_query_with_no_content_words_returns_nothing(pool: DictPool, corpus: str) -> None:
    assert search_lexical(pool, "what is the", QUERY_VECTOR, top_k=3) == []


def test_hybrid_surfaces_the_lexical_hit_that_dense_search_missed(
    pool: DictPool, corpus: str
) -> None:
    hits = search_hybrid(pool, IDENTIFIER_QUERY, QUERY_VECTOR, top_k=3)
    assert "ADR-0002" in hits[0].text
    assert set(hits[0].matched_by) == {"semantic", "lexical"}


def test_hybrid_keeps_strong_semantic_results_too(pool: DictPool, corpus: str) -> None:
    hits = search_hybrid(pool, IDENTIFIER_QUERY, QUERY_VECTOR, top_k=5)
    assert "General guidance about payments and transfers." in texts(hits)


def test_hybrid_order_is_deterministic(pool: DictPool, corpus: str) -> None:
    runs = [texts(search_hybrid(pool, IDENTIFIER_QUERY, QUERY_VECTOR, top_k=5)) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_hybrid_respects_top_k(pool: DictPool, corpus: str) -> None:
    assert len(search_hybrid(pool, IDENTIFIER_QUERY, QUERY_VECTOR, top_k=2)) == 2
    assert len(search_hybrid(pool, IDENTIFIER_QUERY, QUERY_VECTOR, top_k=20)) <= 5
    with pytest.raises(ValueError, match="top_k"):
        search_hybrid(pool, IDENTIFIER_QUERY, QUERY_VECTOR, top_k=0)


def test_hybrid_fuses_scores_from_both_retrievers(pool: DictPool, corpus: str) -> None:
    hits = search_hybrid(pool, IDENTIFIER_QUERY, QUERY_VECTOR, top_k=5, rrf_k=60)
    scores = [h.score for h in hits]
    assert all(s is not None for s in scores)
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_metadata_filters_apply_to_lexical_and_hybrid_retrieval(
    pool: DictPool, corpus: str, mode: str
) -> None:
    filters = {"category": "fees"}
    if mode == "lexical":
        hits = search_lexical(pool, "wire transfer limits grace", QUERY_VECTOR, 5, filters)
    else:
        hits = search_hybrid(pool, "wire transfer limits grace", QUERY_VECTOR, 5, filters)
    assert hits
    assert {h.category for h in hits} == {"fees"}
    assert not any("ADR-0002" in h.text for h in hits)  # the payments chunk is filtered out


def test_filters_still_reject_unknown_keys_in_lexical_search(pool: DictPool, corpus: str) -> None:
    with pytest.raises(ValueError, match="cannot filter on"):
        search_lexical(pool, "wire", QUERY_VECTOR, 3, {"text; DROP TABLE chunks": "x"})


def test_query_text_is_data_not_sql(pool: DictPool, corpus: str) -> None:
    search_lexical(pool, "x'); DROP TABLE chunks; --", QUERY_VECTOR, 3)
    assert len(search_chunks(pool, QUERY_VECTOR, 10)) == 5


def test_the_full_text_index_exists_and_can_serve_the_query(corpus: str) -> None:
    with psycopg.connect(corpus) as conn:
        conn.execute("SET enable_seqscan = off")
        plan = "\n".join(
            row[0]
            for row in conn.execute(
                "EXPLAIN SELECT id FROM chunks WHERE tsv @@ to_tsquery('english', 'wire | grace')"
            )
        )
    assert "chunks_tsv_gin" in plan


def test_init_schema_upgrades_a_table_created_before_full_text_search(clean_db: str) -> None:
    with psycopg.connect(clean_db, autocommit=True) as conn:
        conn.execute("DROP INDEX chunks_tsv_gin")
        conn.execute("ALTER TABLE chunks DROP COLUMN tsv")
    seed_chunks(clean_db, [Spec("Wire transfers need callback verification.", basis(0))])
    init_schema(clean_db, TEST_DIM)  # must add the column back and fill it for existing rows
    pool = create_pool(clean_db)
    try:
        assert len(search_lexical(pool, "wire callback", basis(0), 3)) == 1
    finally:
        pool.close()
