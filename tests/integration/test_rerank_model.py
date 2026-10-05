"""The real local cross-encoder, end to end (downloads the ONNX model on first use)."""

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
from app.retrieval.rerank import CrossEncoderReranker
from tests.conftest import TEST_DIM

pytestmark = [pytest.mark.integration, pytest.mark.model]


@pytest.fixture
def client(clean_db: str) -> Iterator[TestClient]:
    settings = Settings(
        database_url=clean_db,
        embedding_dim=TEST_DIM,
        rerank_enabled=True,
        _env_file=None,  # type: ignore[call-arg]
    )
    embedder = FastEmbedProvider(settings.embedding_model)
    pool = create_pool(clean_db)
    try:
        docs = load_documents(Path("sample_data/policies"))
        ingest_documents(pool, embedder, docs, settings.chunk_size, settings.chunk_overlap)
    finally:
        pool.close()
    with TestClient(create_app(settings, embedder)) as client:
        yield client


def test_the_real_cross_encoder_scores_a_relevant_passage_above_an_irrelevant_one() -> None:
    scores = CrossEncoderReranker().score(
        "What is the late payment fee?",
        ["Cats are small domesticated mammals.", "The late fee is the lesser of $35 or 5 percent."],
    )
    assert scores[1] > scores[0]


def test_search_with_reranking_returns_scored_results(client: TestClient) -> None:
    payload = {"query": "What is the late payment fee?", "top_k": 3}
    body = client.post("/api/search", json=payload).json()
    assert body["rerank"] is True
    assert body["results"][0]["section"] == "Late payment fee"
    assert all(h["rerank_score"] is not None for h in body["results"])


def test_ask_still_answers_and_refuses_with_reranking_on(client: TestClient) -> None:
    ok = client.post("/api/ask", json={"question": "What is the late payment fee?"}).json()
    assert ok["status"] == "answered"
    assert ok["sources"][0]["section"] == "Late payment fee"
    no = client.post("/api/ask", json={"question": "How do I bake sourdough bread?"}).json()
    assert no["status"] == "refused"
    assert no["sources"] == []
