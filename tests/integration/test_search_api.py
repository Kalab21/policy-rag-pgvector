"""POST /api/search end to end: FastAPI -> embedder -> pgvector.

The deterministic HashingEmbedder keeps these fast; storage and search are still real
PostgreSQL + pgvector. Real-model relevance is covered in test_semantic_search.py.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.pool import create_pool
from app.ingestion.loader import load_documents
from app.ingestion.pipeline import ingest_documents
from app.main import create_app
from tests.conftest import TEST_DIM
from tests.fakes import HashingEmbedder

pytestmark = pytest.mark.integration


@pytest.fixture
def client(clean_db: str) -> Iterator[TestClient]:
    settings = Settings(database_url=clean_db, embedding_dim=TEST_DIM, _env_file=None)  # type: ignore[call-arg]
    embedder = HashingEmbedder(TEST_DIM)
    pool = create_pool(clean_db)
    try:
        docs = load_documents(Path("sample_data/policies"))
        ingest_documents(pool, embedder, docs, 400, 80)
    finally:
        pool.close()
    with TestClient(create_app(settings, embedder)) as client:
        yield client


def test_search_returns_ranked_hits_with_source_metadata_and_scores(client: TestClient) -> None:
    response = client.post("/api/search", json={"query": "late payment fee", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["metric"] == "cosine"
    assert body["top_k"] == 3
    assert body["embedding_model"]
    hits = body["results"]
    assert 1 <= len(hits) <= 3
    for hit in hits:
        assert {"chunk_id", "document", "title", "version", "category", "section", "text"} <= set(
            hit
        )
        assert hit["similarity"] == pytest.approx(1 - hit["distance"])
    assert [h["distance"] for h in hits] == sorted(h["distance"] for h in hits)


def test_filters_are_applied_in_the_database_query(client: TestClient) -> None:
    response = client.post(
        "/api/search",
        json={"query": "credit score", "top_k": 20, "filters": {"category": "underwriting"}},
    )
    hits = response.json()["results"]
    assert hits
    assert {h["category"] for h in hits} == {"underwriting"}
    assert response.json()["filters"] == {"category": "underwriting"}


def test_status_filter_excludes_superseded_documents(client: TestClient) -> None:
    response = client.post(
        "/api/search",
        json={"query": "minimum credit score", "top_k": 20, "filters": {"status": "current"}},
    )
    assert {h["status"] for h in response.json()["results"]} == {"current"}


def test_a_filter_with_no_matches_gives_an_empty_result_not_an_error(client: TestClient) -> None:
    response = client.post(
        "/api/search", json={"query": "fees", "filters": {"category": "nonexistent"}}
    )
    assert response.status_code == 200
    assert response.json()["results"] == []


@pytest.mark.parametrize("mode", ["semantic", "lexical", "hybrid"])
def test_every_retrieval_mode_is_available_per_request(client: TestClient, mode: str) -> None:
    response = client.post(
        "/api/search", json={"query": "late payment fee grace period", "top_k": 3, "mode": mode}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == mode
    assert body["results"]
    if mode == "hybrid":
        assert all(h["score"] is not None and h["matched_by"] for h in body["results"])


def test_the_default_mode_is_reported(client: TestClient) -> None:
    body = client.post("/api/search", json={"query": "late fee"}).json()
    assert body["mode"] == "semantic"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 1001},
        {"query": "fees", "top_k": 0},
        {"query": "fees", "top_k": 21},
        {"query": "fees", "top_k": "many"},
        {"query": "fees", "filters": {"colour": "red"}},
        {"query": "fees", "filters": {"status": "draft"}},
        {"query": "fees", "unexpected": 1},
        {"query": "fees", "mode": "fuzzy"},
    ],
)
def test_invalid_requests_are_rejected_with_422(
    client: TestClient, payload: dict[str, object]
) -> None:
    assert client.post("/api/search", json=payload).status_code == 422
