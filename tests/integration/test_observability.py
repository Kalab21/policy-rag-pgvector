"""Traces, metrics, logs and request ids through the real stack (PostgreSQL + pgvector).

The embedder is the deterministic HashingEmbedder and the generator is extractive or scripted;
the retrieval, the database and the LangGraph flow are real. The telemetry exporters are
in-memory test doubles (the true external boundary).
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from fastapi.testclient import TestClient
from mcp.client import Client
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.core.config import Settings
from app.db.pool import create_pool
from app.ingestion.loader import load_documents
from app.ingestion.pipeline import ingest_documents
from app.main import create_app
from app.mcp_server.server import build_server
from app.mcp_server.tools import PolicyTools
from app.observability.telemetry import Telemetry, build_telemetry, set_current
from app.rag.generator import GeneratorUnavailableError
from app.services import open_services
from tests.conftest import TEST_DIM
from tests.fakes import HashingEmbedder, KeywordReranker
from tests.unit.test_observability import metric_total

pytestmark = pytest.mark.integration

SECRET_QUESTION = "my password is hunter2 and the fee schedule says what about the late fee?"


class Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@pytest.fixture
def otel() -> Iterator[tuple[Telemetry, InMemorySpanExporter, InMemoryMetricReader]]:
    exporter = InMemorySpanExporter()
    reader = InMemoryMetricReader()
    telemetry = build_telemetry(
        span_exporters=[exporter], metric_readers=[reader], simple_spans=True
    )
    yield telemetry, exporter, reader
    set_current(Telemetry())


@pytest.fixture
def logs() -> Iterator[Capture]:
    from app.observability.logs import JsonFormatter

    handler = Capture()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("policy_rag")
    logger.addHandler(handler)
    yield handler
    logger.removeHandler(handler)


def settings_for(url: str, **kw: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": url,
        "embedding_dim": TEST_DIM,
        "evidence_min_similarity": 0.0,
        **kw,
    }
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def ingest(url: str) -> None:
    pool = create_pool(url)
    try:
        docs = load_documents(Path("sample_data/policies"))
        ingest_documents(pool, HashingEmbedder(TEST_DIM), docs, 400, 80)
    finally:
        pool.close()


@pytest.fixture
def api(clean_db: str, otel, logs) -> Iterator[tuple[TestClient, Any]]:  # type: ignore[no-untyped-def]
    telemetry, exporter, reader = otel
    ingest(clean_db)
    settings = settings_for(clean_db)
    app = create_app(
        settings, HashingEmbedder(TEST_DIM), reranker=KeywordReranker(), telemetry=telemetry
    )
    with TestClient(app) as client:
        yield client, (telemetry, exporter, reader)


def span_map(exporter: InMemorySpanExporter) -> dict[str, Any]:
    return {s.name: s for s in exporter.get_finished_spans()}


def attrs(span: Any) -> dict[str, Any]:
    return dict(span.attributes or {})


# --- traces ------------------------------------------------------------------------------------


def test_an_ask_produces_one_trace_with_the_pipeline_stages(api) -> None:  # type: ignore[no-untyped-def]
    client, (_, exporter, _) = api
    assert (
        client.post("/api/ask", json={"question": "What is the late payment fee?"}).status_code
        == 200
    )
    spans = span_map(exporter)
    expected = {
        "http.request",
        "rag.ask",
        "embedding.query",
        "retrieval.semantic",
        "evidence.assess",
        "generator.generate",
        "citation.validate",
    }
    assert expected <= set(spans)
    assert len({s.context.trace_id for s in spans.values()}) == 1  # a single trace
    assert spans["rag.ask"].parent.span_id == spans["http.request"].context.span_id
    for child in (
        "embedding.query",
        "retrieval.semantic",
        "evidence.assess",
        "generator.generate",
        "citation.validate",
    ):
        assert spans[child].parent is not None
    assert spans["retrieval.semantic"].parent.span_id == spans["rag.ask"].context.span_id


def test_the_spans_carry_the_useful_safe_attributes(api) -> None:  # type: ignore[no-untyped-def]
    client, (_, exporter, _) = api
    client.post("/api/ask", json={"question": "What is the late payment fee?", "top_k": 4})
    spans = span_map(exporter)
    ask = attrs(spans["rag.ask"])
    assert ask["top_k"] == 4
    assert ask["retrieval.mode"] == "semantic"
    assert ask["generator"] == "extractive"
    assert ask["answer.status"] in {"answered", "refused"}
    assert ask["retrieved.count"] >= 1
    assert "best_similarity" in ask
    assert ask["threshold"] == 0.0
    evidence = attrs(spans["evidence.assess"])
    assert {"evidence.status", "chunks.considered", "chunks.used", "threshold"} <= set(evidence)
    assert attrs(spans["generator.generate"])["provider"] == "extractive"
    assert attrs(spans["http.request"])["http.route"] == "/api/ask"
    assert attrs(spans["http.request"])["http.status_code"] == 200


def test_hybrid_search_has_fusion_and_both_retriever_spans(api) -> None:  # type: ignore[no-untyped-def]
    client, (_, exporter, _) = api
    client.post("/api/search", json={"query": "late payment fee", "mode": "hybrid"})
    spans = span_map(exporter)
    assert {
        "retrieval.hybrid",
        "retrieval.semantic",
        "retrieval.lexical",
        "retrieval.fusion",
    } <= set(spans)
    assert spans["retrieval.fusion"].parent.span_id == spans["retrieval.hybrid"].context.span_id


def test_reranking_has_its_own_span_with_the_candidate_count(api) -> None:  # type: ignore[no-untyped-def]
    client, (_, exporter, reader) = api
    client.post("/api/search", json={"query": "late payment fee", "rerank": True, "top_k": 2})
    rerank = attrs(span_map(exporter)["retrieval.rerank"])
    assert rerank["top_k"] == 2
    assert rerank["candidates"] >= 2
    assert rerank["returned"] == 2
    assert metric_total(reader, "policy_rag.rerank.duration") > 0


# --- what must NOT be recorded -----------------------------------------------------------------


def test_the_question_and_document_text_never_reach_spans_or_logs(api, logs: Capture) -> None:  # type: ignore[no-untyped-def]
    client, (_, exporter, _) = api
    client.post("/api/ask", json={"question": SECRET_QUESTION})
    everything = json.dumps([s.to_json() for s in exporter.get_finished_spans()]) + "\n".join(
        logs.lines
    )
    assert "hunter2" not in everything
    assert "fee schedule says" not in everything
    assert "The late fee is" not in everything  # chunk text
    ask = attrs(span_map(exporter)["rag.ask"])
    assert ask["query.length"] == len(SECRET_QUESTION)
    assert "query.preview" not in ask


def test_a_preview_of_the_question_appears_only_when_explicitly_enabled(
    clean_db: str, logs: Capture
) -> None:
    exporter = InMemorySpanExporter()
    telemetry = build_telemetry(
        span_exporters=[exporter],
        metric_readers=[InMemoryMetricReader()],
        simple_spans=True,
        record_query_text=True,
    )
    ingest(clean_db)
    app = create_app(settings_for(clean_db), HashingEmbedder(TEST_DIM), telemetry=telemetry)
    try:
        with TestClient(app) as client:
            client.post("/api/ask", json={"question": "What is the late payment fee?"})
    finally:
        set_current(Telemetry())
    ask = attrs(span_map(exporter)["rag.ask"])
    assert ask["query.preview"] == "What is the late payment fee?"


def test_credentials_in_a_request_header_are_not_recorded(api, logs: Capture) -> None:  # type: ignore[no-untyped-def]
    client, (_, exporter, _) = api
    client.post(
        "/api/search",
        json={"query": "late fee"},
        headers={
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdef",
            "X-Api-Key": "sk-abcdefghijklmnopqrstuv",
        },
    )
    everything = json.dumps([s.to_json() for s in exporter.get_finished_spans()]) + "\n".join(
        logs.lines
    )
    assert "eyJhbGci" not in everything
    assert "sk-abcdef" not in everything


# --- metrics -----------------------------------------------------------------------------------


def test_request_ask_and_latency_metrics_are_recorded(api) -> None:  # type: ignore[no-untyped-def]
    client, (_, _, reader) = api
    client.post("/api/ask", json={"question": "What is the late payment fee?"})
    client.get("/health")
    assert metric_total(reader, "policy_rag.requests", route="/api/ask", status_class="2xx") == 1
    assert metric_total(reader, "policy_rag.requests", route="/health") == 1
    assert metric_total(reader, "policy_rag.ask.total") == 1
    assert metric_total(reader, "policy_rag.retrieval.duration", mode="semantic") > 0
    assert metric_total(reader, "policy_rag.generation.duration", provider="extractive") > 0
    assert metric_total(reader, "policy_rag.retrieval.returned_chunks") >= 1


def test_refusals_are_counted_with_their_reason(clean_db: str, otel, logs) -> None:  # type: ignore[no-untyped-def]
    telemetry, exporter, reader = otel
    ingest(clean_db)
    settings = settings_for(clean_db, evidence_min_similarity=1.0)
    app = create_app(settings, HashingEmbedder(TEST_DIM), telemetry=telemetry)
    with TestClient(app) as client:
        body = client.post("/api/ask", json={"question": "What is the late payment fee?"}).json()
    assert body["status"] == "refused"
    assert metric_total(reader, "policy_rag.ask.total", status="refused") == 1
    assert metric_total(reader, "policy_rag.ask.refusals", reason="insufficient_evidence") == 1
    assert attrs(span_map(exporter)["rag.ask"])["refusal_reason"] == "insufficient_evidence"


def test_a_generator_outage_is_a_502_a_provider_error_and_an_error_span(
    clean_db: str, otel, logs
) -> None:  # type: ignore[no-untyped-def]
    from tests.unit.test_rag_graph import ScriptedGenerator

    class Down(ScriptedGenerator):
        name = "bedrock:test-model"

        def generate(self, question, chunks):  # type: ignore[no-untyped-def]
            raise GeneratorUnavailableError("the Bedrock request timed out")

    telemetry, exporter, reader = otel
    ingest(clean_db)
    app = create_app(
        settings_for(clean_db), HashingEmbedder(TEST_DIM), Down(""), telemetry=telemetry
    )
    with TestClient(app) as client:
        response = client.post("/api/ask", json={"question": "What is the late payment fee?"})
    assert response.status_code == 502
    assert metric_total(reader, "policy_rag.provider.errors", provider="bedrock") == 1
    assert metric_total(reader, "policy_rag.errors", error_type="http_502") == 1
    generate = span_map(exporter)["generator.generate"]
    assert attrs(generate)["error.type"] == "GeneratorUnavailableError"
    assert "timed out" not in (generate.status.description or "")  # type only, never the message


def test_prometheus_text_is_served_at_metrics(clean_db: str, logs) -> None:  # type: ignore[no-untyped-def]
    ingest(clean_db)
    app = create_app(settings_for(clean_db), HashingEmbedder(TEST_DIM))
    try:
        with TestClient(app) as client:
            client.post("/api/ask", json={"question": "What is the late payment fee?"})
            text = client.get("/metrics").text
    finally:
        set_current(Telemetry())
    for name in (
        "policy_rag_requests_total",
        "policy_rag_ask_total",
        "policy_rag_retrieval_duration",
    ):
        assert name in text
    assert "late payment" not in text.lower()


def test_metrics_can_be_switched_off(clean_db: str) -> None:
    ingest(clean_db)
    app = create_app(settings_for(clean_db, metrics_enabled=False), HashingEmbedder(TEST_DIM))
    try:
        with TestClient(app) as client:
            assert client.get("/metrics").status_code == 404
            assert client.get("/health").status_code == 200
    finally:
        set_current(Telemetry())


def test_telemetry_can_be_disabled_without_breaking_the_api(clean_db: str) -> None:
    ingest(clean_db)
    app = create_app(settings_for(clean_db, telemetry_enabled=False), HashingEmbedder(TEST_DIM))
    try:
        with TestClient(app) as client:
            assert client.post("/api/ask", json={"question": "late fee?"}).status_code == 200
            assert client.get("/metrics").status_code == 404
    finally:
        set_current(Telemetry())


# --- request ids and structured logs -----------------------------------------------------------


def test_a_valid_request_id_is_echoed_and_an_invalid_one_replaced(api) -> None:  # type: ignore[no-untyped-def]
    client, _ = api
    assert (
        client.get("/health", headers={"X-Request-ID": "req-abc.123"}).headers["x-request-id"]
        == "req-abc.123"
    )
    bad = client.get("/health", headers={"X-Request-ID": "has spaces; and <script>"})
    assert bad.headers["x-request-id"] != "has spaces; and <script>"
    assert len(bad.headers["x-request-id"]) == 32
    assert len(client.get("/health").headers["x-request-id"]) == 32


def test_request_logs_are_structured_correlated_and_free_of_user_text(api, logs: Capture) -> None:  # type: ignore[no-untyped-def]
    client, _ = api
    client.post(
        "/api/ask", json={"question": SECRET_QUESTION}, headers={"X-Request-ID": "trace-me-1"}
    )
    events = [json.loads(line) for line in logs.lines]
    done = next(e for e in events if e["event"] == "request.completed")
    assert done["request_id"] == "trace-me-1"
    assert done["route"] == "/api/ask"
    assert done["status"] == 200
    assert len(done["trace_id"]) == 32
    asked = next(e for e in events if e["event"] == "rag.ask.completed")
    assert asked["request_id"] == "trace-me-1"
    assert {"status", "retrieved", "threshold", "generator", "duration_ms"} <= set(asked)
    assert all("hunter2" not in line for line in logs.lines)


# --- MCP ---------------------------------------------------------------------------------------


def test_mcp_tool_calls_have_a_span_and_success_and_failure_counters(clean_db: str, otel) -> None:  # type: ignore[no-untyped-def]
    telemetry, exporter, reader = otel
    set_current(telemetry)
    embedder = HashingEmbedder(TEST_DIM)
    services = open_services(settings_for(clean_db), embedder)
    try:
        ingest_documents(
            services.pool, embedder, load_documents(Path("sample_data/policies")), 400, 80
        )
        server = build_server(PolicyTools(services.retrieval, services.rag, services.pool))

        async def go() -> None:
            async with Client(server) as client:
                await client.call_tool("search_policy", {"query": "late fee"})
                await client.call_tool("get_policy_document", {"document": "no-such-document"})

        anyio.run(go)
    finally:
        services.close()
    tools = [s for s in exporter.get_finished_spans() if s.name == "mcp.tool"]
    assert {attrs(s)["tool.name"] for s in tools} == {"search_policy", "get_policy_document"}
    assert (
        metric_total(reader, "policy_rag.mcp.tool.calls", tool="search_policy", outcome="ok") == 1
    )
    assert (
        metric_total(
            reader, "policy_rag.mcp.tool.calls", tool="get_policy_document", outcome="error"
        )
        == 1
    )
    assert metric_total(reader, "policy_rag.mcp.tool.failures", tool="get_policy_document") == 1


# --- OTLP export -------------------------------------------------------------------------------


def test_spans_and_metrics_reach_an_otlp_collector_without_the_question(clean_db: str) -> None:
    """A local HTTP receiver stands in for a collector; the OTLP/HTTP exporter is the real one."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received: list[tuple[str, bytes]] = []

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            received.append((self.path, self.rfile.read(length)))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ingest(clean_db)
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        settings = settings_for(clean_db, otel_exporter_otlp_endpoint=endpoint)
        app = create_app(settings, HashingEmbedder(TEST_DIM))
        with TestClient(app) as client:
            client.post("/api/ask", json={"question": SECRET_QUESTION})
        # leaving the TestClient context shuts telemetry down, which flushes both exporters
    finally:
        server.shutdown()
        set_current(Telemetry())
    paths = {path for path, _ in received}
    assert "/v1/traces" in paths
    assert "/v1/metrics" in paths
    payload = b"".join(body for _, body in received)
    assert b"rag.ask" in payload
    assert b"policy_rag.ask.total" in payload
    assert b"hunter2" not in payload
    assert b"fee schedule says" not in payload
