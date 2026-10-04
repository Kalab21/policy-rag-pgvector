"""Vector search and metadata filtering against a real PostgreSQL + pgvector.

The vectors are hand-made so the expected ranking is exact. Nothing about the search is
mocked: ordering, distances, filters and the index are all PostgreSQL's.
"""

import random

import psycopg
import pytest

from app.db.pool import DictPool, create_pool
from app.db.schema import pgvector_version
from app.retrieval.store import (
    FILTERABLE_KEYS,
    InvalidFilterError,
    search_chunks,
    supports_iterative_scan,
    validate_filters,
)
from tests.conftest import TEST_DIM
from tests.helpers import Spec, basis, literal, seed_chunks

pytestmark = pytest.mark.integration

QUERY = basis(0)  # the query points along axis 0


@pytest.fixture
def pool(clean_db: str):  # type: ignore[no-untyped-def]
    pool = create_pool(clean_db)
    yield pool
    pool.close()


@pytest.fixture
def corpus(clean_db: str) -> str:
    """Five chunks at known angles from the query, in three categories and two statuses."""
    seed_chunks(
        clean_db,
        [
            Spec("exact", basis(0), category="fees", document="fees", section="A"),
            Spec(
                "close", basis(0, tilt=0.2), category="underwriting", document="uw", version="2.0"
            ),
            Spec(
                "old-close",
                basis(0, tilt=0.4),
                category="underwriting",
                document="uw",
                version="1.0",
                status="superseded",
            ),
            Spec("far", basis(0, tilt=3.0), category="privacy", document="priv"),
            Spec("orthogonal", basis(1), category="fees", document="fees", section="B"),
        ],
    )
    return clean_db


def test_results_are_ordered_by_cosine_distance(pool: DictPool, corpus: str) -> None:
    hits = search_chunks(pool, QUERY, top_k=5)
    assert [h.text for h in hits] == ["exact", "close", "old-close", "far", "orthogonal"]
    distances = [h.distance for h in hits]
    assert distances == sorted(distances)


def test_distance_and_similarity_follow_the_cosine_definition(pool: DictPool, corpus: str) -> None:
    hits = {h.text: h for h in search_chunks(pool, QUERY, top_k=5)}
    assert hits["exact"].distance == pytest.approx(0.0, abs=1e-6)
    assert hits["exact"].similarity == pytest.approx(1.0, abs=1e-6)
    assert hits["orthogonal"].distance == pytest.approx(1.0, abs=1e-6)
    # cos(angle) for a vector tilted by 0.2: 1 / sqrt(1 + 0.2^2)
    assert hits["close"].similarity == pytest.approx(1 / (1 + 0.2**2) ** 0.5, abs=1e-5)
    for h in hits.values():
        assert h.similarity == pytest.approx(1.0 - h.distance, abs=1e-9)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_top_k_limits_the_number_of_results(pool: DictPool, corpus: str, k: int) -> None:
    hits = search_chunks(pool, QUERY, top_k=k)
    assert len(hits) == k
    assert hits[0].text == "exact"


def test_top_k_larger_than_the_corpus_returns_everything(pool: DictPool, corpus: str) -> None:
    assert len(search_chunks(pool, QUERY, top_k=20)) == 5


def test_top_k_must_be_positive(pool: DictPool, corpus: str) -> None:
    with pytest.raises(ValueError, match="top_k"):
        search_chunks(pool, QUERY, top_k=0)


def test_a_category_filter_returns_only_that_category(pool: DictPool, corpus: str) -> None:
    hits = search_chunks(pool, QUERY, top_k=5, filters={"category": "underwriting"})
    assert [h.text for h in hits] == ["close", "old-close"]
    assert {h.category for h in hits} == {"underwriting"}


def test_the_filter_excludes_chunks_even_when_they_are_the_closest(
    pool: DictPool, corpus: str
) -> None:
    unfiltered = search_chunks(pool, QUERY, top_k=1)
    filtered = search_chunks(pool, QUERY, top_k=1, filters={"category": "privacy"})
    assert unfiltered[0].text == "exact"
    assert filtered[0].text == "far"


def test_a_status_filter_hides_superseded_versions(pool: DictPool, corpus: str) -> None:
    hits = search_chunks(pool, QUERY, top_k=5, filters={"status": "current"})
    assert "old-close" not in [h.text for h in hits]
    assert {h.status for h in hits} == {"current"}


def test_a_version_filter_selects_one_version_of_a_document(pool: DictPool, corpus: str) -> None:
    hits = search_chunks(pool, QUERY, top_k=5, filters={"document": "uw", "version": "1.0"})
    assert [h.text for h in hits] == ["old-close"]


def test_filters_combine_with_and(pool: DictPool, corpus: str) -> None:
    hits = search_chunks(
        pool, QUERY, top_k=5, filters={"category": "underwriting", "status": "current"}
    )
    assert [h.text for h in hits] == ["close"]
    hits = search_chunks(pool, QUERY, top_k=5, filters={"category": "fees", "section": "B"})
    assert [h.text for h in hits] == ["orthogonal"]


def test_a_filter_that_matches_nothing_returns_an_empty_list(pool: DictPool, corpus: str) -> None:
    assert search_chunks(pool, QUERY, top_k=5, filters={"category": "does-not-exist"}) == []


def test_top_k_applies_after_filtering(pool: DictPool, corpus: str) -> None:
    hits = search_chunks(pool, QUERY, top_k=5, filters={"category": "fees"})
    assert [h.text for h in hits] == ["exact", "orthogonal"]


def test_unknown_filter_keys_are_rejected_before_any_sql(pool: DictPool, corpus: str) -> None:
    with pytest.raises(InvalidFilterError, match="cannot filter on"):
        search_chunks(pool, QUERY, top_k=3, filters={"metadata->x": "1"})
    assert set(validate_filters(dict.fromkeys(FILTERABLE_KEYS, "v"))) == set(FILTERABLE_KEYS)


def test_filter_values_are_data_not_sql(pool: DictPool, corpus: str) -> None:
    hits = search_chunks(pool, QUERY, top_k=5, filters={"category": "x'; DROP TABLE chunks; --"})
    assert hits == []
    assert len(search_chunks(pool, QUERY, top_k=5)) == 5  # the table is intact


def test_the_hnsw_index_can_serve_the_query_shape(corpus: str) -> None:
    """With sequential scans disabled the planner must pick the HNSW index, which shows the
    `ORDER BY embedding <=> query LIMIT k` shape is index-eligible. (On a handful of rows
    PostgreSQL normally prefers an exact scan, which is also correct.)"""
    with psycopg.connect(corpus) as conn:
        conn.execute("SET enable_seqscan = off")
        plan = "\n".join(
            row[0]
            for row in conn.execute(
                "EXPLAIN SELECT id FROM chunks"
                f" ORDER BY embedding <=> '{literal(QUERY)}'::vector LIMIT 3"
            )
        )
    assert "chunks_embedding_hnsw" in plan


def test_a_selective_filter_still_fills_top_k_with_an_indexed_scan(clean_db: str) -> None:
    """HNSW returns candidates, then the filter is applied; a rare category can leave fewer
    than k rows. pgvector 0.8 iterative scan keeps scanning until k rows pass the filter."""
    with psycopg.connect(clean_db) as conn:
        version = pgvector_version(conn)
    if not supports_iterative_scan(version):
        pytest.skip(f"iterative scan needs pgvector 0.8+, found {version}")

    rng = random.Random(7)
    specs = []
    for i in range(400):
        vector = [rng.gauss(0, 1) for _ in range(TEST_DIM)]
        category = "rare" if i % 80 == 0 else "common"
        specs.append(Spec(f"c{i}", vector, category=category, document=f"d{i % 5}"))
    seed_chunks(clean_db, specs)

    query = [rng.gauss(0, 1) for _ in range(TEST_DIM)]
    sql = (
        'SELECT id FROM chunks WHERE metadata @> \'{"category": "rare"}\''
        f" ORDER BY embedding <=> '{literal(query)}'::vector LIMIT 5"
    )
    with psycopg.connect(clean_db) as conn:
        conn.execute("SET enable_seqscan = off")
        conn.execute("SET hnsw.ef_search = 40")
        conn.execute("SET hnsw.iterative_scan = strict_order")
        rows = conn.execute(sql).fetchall()
    assert len(rows) == 5  # exactly the five "rare" chunks, found through the index
