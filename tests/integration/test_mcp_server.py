"""The MCP server through the real protocol (in-process client), on real PostgreSQL + pgvector.

Retrieval and the database are real. The embedder is the deterministic HashingEmbedder (and the
generator is extractive), so these run fast; the stdio smoke test at the end uses the real model.
"""

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult

from app.core.config import Settings
from app.db.pool import create_pool
from app.embeddings.fastembed_provider import FastEmbedProvider
from app.ingestion.loader import load_documents
from app.ingestion.pipeline import ingest_documents
from app.mcp_server.server import build_server
from app.mcp_server.tools import PolicyTools
from app.services import Services, open_services
from tests.conftest import TEST_DIM
from tests.fakes import HashingEmbedder, KeywordReranker

pytestmark = pytest.mark.integration


@pytest.fixture
def services(clean_db: str) -> Iterator[Services]:
    settings = Settings(
        database_url=clean_db,
        embedding_dim=TEST_DIM,
        evidence_min_similarity=0.0,  # the HashingEmbedder is not calibrated for the real gate
        _env_file=None,  # type: ignore[call-arg]
    )
    embedder = HashingEmbedder(TEST_DIM)
    svc = open_services(settings, embedder, reranker=KeywordReranker())
    ingest_documents(svc.pool, embedder, load_documents(Path("sample_data/policies")), 400, 80)
    yield svc
    svc.close()


def call(
    svc: Services,
    name: str,
    arguments: dict[str, Any],
    *,
    enforced: dict[str, str] | None = None,
    timeout_s: float = 30.0,
    tools: PolicyTools | None = None,
) -> CallToolResult:
    server = build_server(
        tools or PolicyTools(svc.retrieval, svc.rag, svc.pool, enforced), timeout_s
    )

    async def go() -> CallToolResult:
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return anyio.run(go)


def text_of(result: CallToolResult) -> str:
    return " ".join(getattr(block, "text", "") for block in result.content)


def structured(result: CallToolResult) -> dict[str, Any]:
    assert not result.is_error, text_of(result)
    assert result.structured_content is not None
    return result.structured_content


def counts(svc: Services) -> tuple[int, int]:
    with svc.pool.connection() as conn:
        d = conn.execute("SELECT count(*) AS n FROM documents").fetchone()
        c = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()
    assert d is not None
    assert c is not None
    return int(d["n"]), int(c["n"])


# --- what the server exposes -------------------------------------------------------------------


def test_exactly_three_read_only_tools_are_exposed(services: Services) -> None:
    server = build_server(PolicyTools(services.retrieval, services.rag, services.pool))

    async def go() -> Any:
        async with Client(server) as client:
            return await client.list_tools()

    tools = anyio.run(go).tools
    assert {t.name for t in tools} == {"search_policy", "get_policy_document", "ask_policy"}
    for tool in tools:
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        assert tool.annotations.open_world_hint is False


def test_the_published_schemas_carry_the_input_bounds(services: Services) -> None:
    server = build_server(PolicyTools(services.retrieval, services.rag, services.pool))

    async def go() -> Any:
        async with Client(server) as client:
            return await client.list_tools()

    schemas = {t.name: t.input_schema for t in anyio.run(go).tools}
    search = schemas["search_policy"]["properties"]
    assert search["top_k"]["maximum"] == 10
    assert search["top_k"]["minimum"] == 1
    assert search["query"]["maxLength"] == 1000
    assert "pattern" in schemas["get_policy_document"]["properties"]["document"]


# --- search_policy -----------------------------------------------------------------------------


def test_search_policy_returns_ranked_passages_with_sources(services: Services) -> None:
    body = structured(call(services, "search_policy", {"query": "late payment fee", "top_k": 3}))
    assert 1 <= len(body["results"]) <= 3
    assert {"document", "version", "section", "text", "similarity"} <= set(body["results"][0])
    assert body["mode"] == "semantic"


def test_search_policy_applies_metadata_filters(services: Services) -> None:
    args = {"query": "credit score", "top_k": 10, "filters": {"category": "underwriting"}}
    body = structured(call(services, "search_policy", args))
    assert body["results"]
    assert {h["category"] for h in body["results"]} == {"underwriting"}


@pytest.mark.parametrize("mode", ["semantic", "lexical", "hybrid"])
def test_search_policy_supports_every_retrieval_mode(services: Services, mode: str) -> None:
    args = {"query": "late payment fee grace", "top_k": 3, "mode": mode}
    body = structured(call(services, "search_policy", args))
    assert body["mode"] == mode
    assert body["results"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 1001},
        {"query": "fees", "top_k": 0},
        {"query": "fees", "top_k": 11},
        {"query": "fees", "top_k": "many"},
        {"query": "fees", "mode": "fuzzy"},
        {"query": "fees", "filters": {"colour": "red"}},
        {"query": "fees", "filters": {"status": "draft"}},
        {"query": "fees", "filters": {"category": "x" * 201}},
        {},
    ],
)
def test_invalid_search_arguments_are_rejected_as_tool_errors(
    services: Services, arguments: dict[str, Any]
) -> None:
    result = call(services, "search_policy", arguments)
    assert result.is_error
    assert services.retrieval is not None


def test_undeclared_arguments_are_ignored_and_never_reach_any_code(services: Services) -> None:
    """The SDK drops arguments the tool does not declare, so an extra 'sql' field does nothing."""
    before = counts(services)
    plain = structured(call(services, "search_policy", {"query": "late payment fee", "top_k": 2}))
    extra = structured(
        call(
            services,
            "search_policy",
            {"query": "late payment fee", "top_k": 2, "sql": "DELETE FROM chunks", "path": "/etc"},
        )
    )
    assert [h["chunk_id"] for h in extra["results"]] == [h["chunk_id"] for h in plain["results"]]
    assert counts(services) == before


def test_a_query_that_looks_like_sql_is_just_text(services: Services) -> None:
    before = counts(services)
    result = call(services, "search_policy", {"query": "x'); DROP TABLE chunks; --"})
    assert not result.is_error
    assert counts(services) == before


# --- get_policy_document -----------------------------------------------------------------------


def test_get_policy_document_defaults_to_the_current_version(services: Services) -> None:
    body = structured(call(services, "get_policy_document", {"document": "underwriting-policy"}))
    assert body["version"] == "2.0"
    assert body["status"] == "current"
    assert {p["section"] for p in body["passages"]} >= {"Credit score decision bands"}


def test_get_policy_document_can_read_a_superseded_version(services: Services) -> None:
    args = {"document": "underwriting-policy", "version": "1.0"}
    body = structured(call(services, "get_policy_document", args))
    assert (body["version"], body["status"]) == ("1.0", "superseded")


def test_an_unknown_document_is_a_safe_error(services: Services) -> None:
    result = call(services, "get_policy_document", {"document": "no-such-document"})
    assert result.is_error
    assert "no such document" in text_of(result)


@pytest.mark.parametrize(
    "document", ["../../etc/passwd", "/etc/passwd", "C:\\boot.ini", "x'; DROP TABLE documents; --"]
)
def test_paths_and_sql_cannot_be_passed_as_a_document_name(
    services: Services, document: str
) -> None:
    before = counts(services)
    result = call(services, "get_policy_document", {"document": document})
    assert result.is_error
    assert "passwd" not in text_of(result).replace(document, "")  # nothing was read
    assert counts(services) == before


# --- ask_policy --------------------------------------------------------------------------------


def test_ask_policy_returns_a_cited_answer_or_a_refusal_with_the_same_shape(
    services: Services,
) -> None:
    body = structured(call(services, "ask_policy", {"question": "What is the late payment fee?"}))
    assert body["status"] in {"answered", "refused"}
    assert {"answer", "sources", "evidence", "retrieved_chunk_ids"} <= set(body)
    if body["status"] == "answered":
        assert body["sources"]
        assert {s["chunk_id"] for s in body["sources"]} <= set(body["retrieved_chunk_ids"])


def test_ask_policy_refuses_when_evidence_is_insufficient(clean_db: str) -> None:
    settings = Settings(
        database_url=clean_db,
        embedding_dim=TEST_DIM,
        evidence_min_similarity=1.0,  # nothing can reach a similarity of exactly 1.0
        _env_file=None,  # type: ignore[call-arg]
    )
    embedder = HashingEmbedder(TEST_DIM)
    svc = open_services(settings, embedder, reranker=KeywordReranker())
    try:
        ingest_documents(svc.pool, embedder, load_documents(Path("sample_data/policies")), 400, 80)
        body = structured(call(svc, "ask_policy", {"question": "What is the late payment fee?"}))
    finally:
        svc.close()
    assert body["status"] == "refused"
    assert body["sources"] == []
    assert body["evidence"]["status"] == "insufficient"


def test_ask_policy_validates_its_arguments(services: Services) -> None:
    assert call(services, "ask_policy", {"question": ""}).is_error
    assert call(services, "ask_policy", {"question": "q", "filters": {"x": "y"}}).is_error


# --- failures ----------------------------------------------------------------------------------


class ExplodingRetrieval:
    model_name = "stub"
    mode = "semantic"
    rerank_enabled = False

    def search(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("password=hunter2 host=internal-db")


def test_unexpected_failures_do_not_leak_internals(services: Services) -> None:
    tools = PolicyTools(ExplodingRetrieval(), services.rag, services.pool)  # type: ignore[arg-type]
    result = call(services, "search_policy", {"query": "fees"}, tools=tools)
    assert result.is_error
    assert "hunter2" not in text_of(result)
    assert "internal-db" not in text_of(result)


class SlowRetrieval(ExplodingRetrieval):
    def search(self, *args: Any, **kwargs: Any) -> Any:
        import time

        time.sleep(2)
        return []


def test_a_slow_tool_call_times_out_with_a_clear_error(services: Services) -> None:
    tools = PolicyTools(SlowRetrieval(), services.rag, services.pool)  # type: ignore[arg-type]
    result = call(services, "search_policy", {"query": "fees"}, tools=tools, timeout_s=0.2)
    assert result.is_error
    assert "timed out" in text_of(result)


def test_calling_a_tool_that_does_not_exist_is_an_error(services: Services) -> None:
    assert call(services, "run_sql", {"sql": "SELECT 1"}).is_error
    assert call(services, "read_file", {"path": "/etc/passwd"}).is_error


def test_no_tool_changes_the_database(services: Services) -> None:
    before = counts(services)
    call(services, "search_policy", {"query": "late fee"})
    call(services, "get_policy_document", {"document": "fee-schedule"})
    call(services, "ask_policy", {"question": "What is the late fee?"})
    assert counts(services) == before


# --- the authorization boundary ---------------------------------------------------------------


def test_enforced_filters_limit_every_tool(services: Services) -> None:
    enforced = {"category": "fees"}
    found = structured(
        call(services, "search_policy", {"query": "records retention"}, enforced=enforced)
    )
    assert found["results"]
    assert {h["category"] for h in found["results"]} == {"fees"}

    asked = structured(
        call(services, "ask_policy", {"question": "How long are records kept?"}, enforced=enforced)
    )
    assert all(s["category"] == "fees" for s in asked["sources"])

    # A document outside the enforced scope is indistinguishable from one that does not exist.
    blocked = call(
        services,
        "get_policy_document",
        {"document": "data-privacy-and-retention"},
        enforced=enforced,
    )
    assert blocked.is_error
    assert "no such document" in text_of(blocked)
    allowed = call(services, "get_policy_document", {"document": "fee-schedule"}, enforced=enforced)
    assert not allowed.is_error


def test_a_caller_cannot_widen_or_change_an_enforced_filter(services: Services) -> None:
    enforced = {"category": "fees"}
    args = {"query": "retention", "filters": {"category": "privacy"}}
    result = call(services, "search_policy", args, enforced=enforced)
    assert result.is_error
    assert "fixed by the server" in text_of(result)
    assert call(
        services,
        "ask_policy",
        {"question": "q", "filters": {"category": "privacy"}},
        enforced=enforced,
    ).is_error


# --- a real MCP client over stdio ---------------------------------------------------------------


@pytest.mark.model
def test_the_server_works_over_stdio_with_the_real_model(clean_db: str) -> None:
    """Launches `python -m app.mcp_server` as a subprocess and talks to it as an MCP client."""
    settings = Settings(database_url=clean_db, embedding_dim=TEST_DIM, _env_file=None)  # type: ignore[call-arg]
    embedder = FastEmbedProvider(settings.embedding_model)
    pool = create_pool(clean_db)
    try:
        docs = load_documents(Path("sample_data/policies"))
        ingest_documents(pool, embedder, docs, settings.chunk_size, settings.chunk_overlap)
    finally:
        pool.close()

    env = {**os.environ, "DATABASE_URL": clean_db, "AUTO_INIT_SCHEMA": "false"}
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "app.mcp_server"], env=env, cwd=Path.cwd()
    )

    async def go() -> tuple[set[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
        async with Client(params) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            found = await client.call_tool(
                "search_policy", {"query": "What is the late payment fee?", "top_k": 2}
            )
            asked = await client.call_tool(
                "ask_policy", {"question": "What is the late payment fee?"}
            )
            refused = await client.call_tool(
                "ask_policy", {"question": "How do I bake sourdough bread?"}
            )
            return (
                tools,
                found.structured_content or {},
                asked.structured_content or {},
                refused.structured_content or {},
            )

    tools, found, asked, refused = anyio.run(go)
    assert tools == {"search_policy", "get_policy_document", "ask_policy"}
    assert found["results"][0]["section"] == "Late payment fee"
    assert asked["status"] == "answered"
    assert asked["sources"][0]["section"] == "Late payment fee"
    assert refused["status"] == "refused"
    assert refused["sources"] == []
