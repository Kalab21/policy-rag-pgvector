"""Relevance with the real embedding model against the bundled synthetic policies."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.pool import create_pool
from app.embeddings.fastembed_provider import FastEmbedProvider
from app.ingestion.loader import load_documents
from app.ingestion.pipeline import ingest_documents
from app.main import create_app
from tests.conftest import TEST_DIM

pytestmark = [pytest.mark.integration, pytest.mark.model]


@pytest.fixture
def client(clean_db: str) -> Iterator[TestClient]:
    settings = Settings(database_url=clean_db, embedding_dim=TEST_DIM, _env_file=None)  # type: ignore[call-arg]
    embedder = FastEmbedProvider(settings.embedding_model)
    pool = create_pool(clean_db)
    try:
        docs = load_documents(Path("sample_data/policies"))
        ingest_documents(pool, embedder, docs, settings.chunk_size, settings.chunk_overlap)
    finally:
        pool.close()
    with TestClient(create_app(settings, embedder)) as client:
        yield client


def _search(client: TestClient, query: str, **filters: str) -> list[dict[str, object]]:
    body: dict[str, object] = {"query": query, "top_k": 3}
    if filters:
        body["filters"] = filters
    response = client.post("/api/search", json=body)
    assert response.status_code == 200
    results: list[dict[str, object]] = response.json()["results"]
    return results


def test_a_fee_question_finds_the_fee_schedule(client: TestClient) -> None:
    hits = _search(client, "What is the late payment fee?")
    assert hits[0]["document"] == "fee-schedule"
    assert "late" in str(hits[0]["text"]).lower()


def test_a_privacy_question_finds_the_privacy_policy(client: TestClient) -> None:
    hits = _search(client, "How long do you keep my records?")
    assert hits[0]["document"] == "data-privacy-and-retention"


def test_status_filter_keeps_superseded_underwriting_rules_out(client: TestClient) -> None:
    query = "minimum credit score for automatic approval"
    current = _search(client, query, status="current")
    assert current
    assert {h["status"] for h in current} == {"current"}
    assert current[0]["version"] == "2.0"
    old = _search(client, query, status="superseded")
    assert old
    assert {h["version"] for h in old} == {"1.0"}
