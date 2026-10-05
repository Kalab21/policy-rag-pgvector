"""POST /api/ask end to end: real PostgreSQL + pgvector, real embedding model."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.pool import create_pool
from app.embeddings.fastembed_provider import FastEmbedProvider
from app.ingestion.loader import load_documents
from app.ingestion.pipeline import ingest_documents
from app.main import create_app
from app.rag.bedrock import BedrockGenerator
from tests.conftest import TEST_DIM
from tests.fakes import FakeBedrockClient, tool_response
from tests.unit.test_rag_graph import ScriptedGenerator

pytestmark = pytest.mark.integration


@pytest.fixture
def ready(clean_db: str) -> tuple[Settings, FastEmbedProvider]:
    settings = Settings(database_url=clean_db, embedding_dim=TEST_DIM, _env_file=None)  # type: ignore[call-arg]
    embedder = FastEmbedProvider(settings.embedding_model)
    pool = create_pool(clean_db)
    try:
        docs = load_documents(Path("sample_data/policies"))
        ingest_documents(pool, embedder, docs, settings.chunk_size, settings.chunk_overlap)
    finally:
        pool.close()
    return settings, embedder


@pytest.fixture
def client(ready: tuple[Settings, FastEmbedProvider]) -> Iterator[TestClient]:
    settings, embedder = ready
    with TestClient(create_app(settings, embedder)) as client:
        yield client


def ask(client: TestClient, question: str, **extra: Any) -> dict[str, Any]:
    response = client.post("/api/ask", json={"question": question, **extra})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


@pytest.mark.model
def test_an_answerable_question_gets_a_cited_answer(client: TestClient) -> None:
    body = ask(client, "What is the late payment fee?")
    assert body["status"] == "answered"
    assert body["evidence"]["status"] == "sufficient"
    assert "$35" in body["answer"]
    assert "[1]" in body["answer"]
    source = body["sources"][0]
    assert source["document"] == "fee-schedule"
    assert source["citation"] == 1
    assert "$35" in source["text"]


@pytest.mark.model
def test_every_cited_source_was_actually_retrieved(client: TestClient) -> None:
    body = ask(client, "How long do you retain customer records?")
    assert body["status"] == "answered"
    assert body["sources"]
    assert {s["chunk_id"] for s in body["sources"]} <= set(body["retrieved_chunk_ids"])
    for source in body["sources"]:
        assert f"[{source['citation']}]" in body["answer"]


@pytest.mark.model
@pytest.mark.parametrize(
    "question",
    [
        "How do I bake sourdough bread?",
        "Who won the 2022 World Cup?",
        "What is the weather in Paris today?",
        "What is the policy on cryptocurrency loans?",
    ],
)
def test_questions_the_documents_cannot_answer_are_refused(
    client: TestClient, question: str
) -> None:
    body = ask(client, question)
    assert body["status"] == "refused"
    assert body["evidence"]["status"] == "insufficient"
    assert body["sources"] == []
    assert body["retrieved_chunk_ids"]  # retrieval still ran; the gate is what said no
    assert body["refusal_reason"] == "insufficient_evidence"


@pytest.mark.model
def test_answers_use_current_policy_by_default_and_superseded_on_request(
    client: TestClient,
) -> None:
    question = "What credit score is needed for automatic approval?"
    current = ask(client, question)
    assert current["status"] == "answered"
    assert {s["version"] for s in current["sources"]} == {"2.0"}
    assert "640" in current["answer"]

    old = ask(client, question, filters={"status": "superseded"})
    assert {s["version"] for s in old["sources"]} == {"1.0"}
    assert "660" in old["answer"]


def test_a_generator_citing_a_source_that_was_not_supplied_is_refused(
    ready: tuple[Settings, FastEmbedProvider],
) -> None:
    settings, embedder = ready
    app = create_app(settings, embedder, ScriptedGenerator("The fee is $99 [9]."))
    with TestClient(app) as client:
        body = ask(client, "What is the late payment fee?")
    assert body["status"] == "refused"
    assert body["refusal_reason"] == "answer_not_grounded_in_sources"
    assert body["sources"] == []


def test_a_generator_with_a_valid_citation_is_accepted(
    ready: tuple[Settings, FastEmbedProvider],
) -> None:
    settings, embedder = ready
    app = create_app(settings, embedder, ScriptedGenerator("Stated in the schedule [1]."))
    with TestClient(app) as client:
        body = ask(client, "What is the late payment fee?")
    assert body["status"] == "answered"
    assert body["generator"] == "scripted"
    assert body["sources"][0]["chunk_id"] in body["retrieved_chunk_ids"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "q", "top_k": 0},
        {"question": "q", "top_k": 99},
        {"question": "q", "filters": {"bogus": "x"}},
        {"question": "q", "extra": True},
    ],
)
def test_invalid_ask_requests_are_rejected_with_422(
    client: TestClient, payload: dict[str, Any]
) -> None:
    assert client.post("/api/ask", json=payload).status_code == 422


def _ask_with_bedrock(
    ready: tuple[Settings, FastEmbedProvider], client: FakeBedrockClient
) -> tuple[int, dict[str, Any]]:
    settings, embedder = ready
    generator = BedrockGenerator("test-model", client=client)
    with TestClient(create_app(settings, embedder, generator)) as api:
        response = api.post("/api/ask", json={"question": "What is the late payment fee?"})
    return response.status_code, response.json()


def test_a_bedrock_answer_flows_through_the_same_gate_and_citation_checks(
    ready: tuple[Settings, FastEmbedProvider],
) -> None:
    client = FakeBedrockClient(
        tool_response({"answer": "The late fee is $35 [1].", "citations": [1]})
    )
    status, body = _ask_with_bedrock(ready, client)
    assert status == 200
    assert body["status"] == "answered"
    assert body["generator"] == "bedrock:test-model"
    assert body["sources"][0]["chunk_id"] in body["retrieved_chunk_ids"]
    # The model only ever saw chunks that passed the evidence gate.
    assert "$35" in client.requests[0]["messages"][0]["content"][0]["text"]


def test_a_bedrock_model_that_finds_no_answer_leads_to_a_refusal(
    ready: tuple[Settings, FastEmbedProvider],
) -> None:
    payload = {"answer": "", "citations": [], "insufficient_evidence": True}
    _, body = _ask_with_bedrock(ready, FakeBedrockClient(tool_response(payload)))
    assert body["status"] == "refused"
    assert body["refusal_reason"] == "generator_found_no_answer"


def test_unvalidated_bedrock_output_is_refused_not_served(
    ready: tuple[Settings, FastEmbedProvider],
) -> None:
    client = FakeBedrockClient(tool_response({"answer": "The fee is $99.", "citations": [7]}))
    status, body = _ask_with_bedrock(ready, client)
    assert status == 200
    assert body["status"] == "refused"
    assert body["refusal_reason"] == "generator_output_invalid"
    assert "$99" not in body["answer"]


def test_a_bedrock_outage_is_a_502_with_a_safe_message(
    ready: tuple[Settings, FastEmbedProvider],
) -> None:
    from botocore.exceptions import ClientError

    secret = "AKIAEXAMPLESECRET1234"
    error = ClientError({"Error": {"Code": "AccessDeniedException", "Message": secret}}, "Converse")
    status, body = _ask_with_bedrock(ready, FakeBedrockClient(error=error))
    assert status == 502
    assert secret not in str(body)
