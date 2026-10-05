"""Authentication and document-level authorization, end to end on real PostgreSQL + pgvector.

The rule under test: what a caller may read is decided in the retrieval SQL, from a validated
token, before ranking, reranking, the evidence gate and generation. So restricted text must
never reach a generator, a reranker, a trace, a log line, or a response.
"""

import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from fastapi.testclient import TestClient
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.core.config import Settings
from app.db.pool import create_pool
from app.embeddings.fastembed_provider import FastEmbedProvider
from app.ingestion.loader import load_documents, parse_document
from app.ingestion.pipeline import ingest_documents
from app.main import create_app
from app.mcp_server.server import build_server
from app.mcp_server.tools import PolicyTools
from app.models.domain import ParsedDocument, RetrievedChunk
from app.observability.logs import JsonFormatter
from app.observability.telemetry import Telemetry, build_telemetry, set_current
from app.security.auth import TokenValidator
from app.services import open_services
from tests import tokens
from tests.conftest import TEST_DIM
from tests.fakes import HashingEmbedder, KeywordReranker
from tests.helpers import Spec, basis, seed_chunks

pytestmark = pytest.mark.integration

# Text that exists only in restricted documents.
OTHER_SECRET = "a-different-secret-that-is-32-chars!!!"
UNDERWRITING = "eligible for automated approval"  # underwriting-policy v2, restricted
PRIVACY = "kept for 7 years"  # data-privacy-and-retention, compliance, restricted
ACME = "ACMEUNIQUEMARKER"  # another tenant's restricted document
LEGAL = "LEGALUNIQUEMARKER"  # a confidential document in the default tenant

EXTRA_DOCS = [
    (
        "acme-underwriting.md",
        f'---\nname: acme-underwriting\ntitle: Acme Underwriting\nversion: "1.0"\ncategory: '
        f"underwriting\nstatus: current\ntenant_id: acme\ndepartment: underwriting\n"
        f"access_level: restricted\n---\n\n## Rules\n{ACME} applies to Acme credit decisions "
        f"and the eligible for automated approval threshold.\n",
    ),
    (
        "legal-hold.md",
        f'---\nname: legal-hold\ntitle: Legal Hold Procedure\nversion: "1.0"\ncategory: legal\n'
        f"status: current\ntenant_id: default\ndepartment: legal\naccess_level: confidential\n"
        f"---\n\n## Hold\n{LEGAL} describes the litigation hold procedure.\n",
    ),
]


class SpyGenerator:
    """Records exactly which chunk texts a generator is handed."""

    name = "spy"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        self.seen.extend(c.text for c in chunks)
        return "Answer from the provided sources [1]."


class Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


def app_settings(url: str, **overrides: Any) -> Settings:
    return tokens.settings(
        url,
        embedding_dim=TEST_DIM,
        evidence_min_similarity=0.0,  # the gate is not under test; the scope is
        **overrides,
    )


def ingest_all(url: str, embedder: Any = None) -> None:
    embedder = embedder or HashingEmbedder(TEST_DIM)
    docs: list[ParsedDocument] = load_documents(Path("sample_data/policies"))
    docs += [parse_document(raw, name) for name, raw in EXTRA_DOCS]
    pool = create_pool(url)
    try:
        ingest_documents(pool, embedder, docs, 400, 80)
    finally:
        pool.close()


def bearer(
    role: str | list[str], *, department: Any = None, tenant: str = "default"
) -> dict[str, str]:
    roles = [role] if isinstance(role, str) else role
    token = tokens.hs256(roles=roles, department=department or [], tenant_id=tenant)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def stack(clean_db: str) -> Iterator[dict[str, Any]]:
    exporter = InMemorySpanExporter()
    telemetry = build_telemetry(
        span_exporters=[exporter], metric_readers=[InMemoryMetricReader()], simple_spans=True
    )
    handler = Capture()
    handler.setFormatter(JsonFormatter())
    logging.getLogger("policy_rag").addHandler(handler)
    ingest_all(clean_db)
    generator, reranker = SpyGenerator(), KeywordReranker()
    app = create_app(
        app_settings(clean_db), HashingEmbedder(TEST_DIM), generator, reranker, telemetry
    )
    with TestClient(app) as client:
        yield {
            "client": client,
            "generator": generator,
            "reranker": reranker,
            "exporter": exporter,
            "logs": handler,
            "url": clean_db,
        }
    logging.getLogger("policy_rag").removeHandler(handler)
    set_current(Telemetry())


def search(
    stack: dict[str, Any], headers: dict[str, str], query: str, **extra: Any
) -> dict[str, Any]:
    body = {"query": query, "top_k": 20, **extra}
    response = stack["client"].post("/api/search", json=body, headers=headers)
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


def texts(body: dict[str, Any]) -> str:
    return " ".join(hit["text"] for hit in body["results"])


def documents_of(body: dict[str, Any]) -> set[str]:
    return {hit["document"] for hit in body["results"]}


# --- authentication ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/api/search", {"query": "fees"}),
        ("post", "/api/ask", {"question": "fees?"}),
        ("get", "/api/documents", None),
        ("get", "/api/me", None),
    ],
)
def test_protected_endpoints_reject_a_missing_token(
    stack: dict[str, Any], method: str, path: str, payload: Any
) -> None:
    client: TestClient = stack["client"]
    response = getattr(client, method)(path, **({"json": payload} if payload else {}))
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert "token" not in response.text.lower() or "credentials" in response.text.lower()


def test_health_stays_open_for_probes(stack: dict[str, Any]) -> None:
    assert stack["client"].get("/health").status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer not-a-jwt"},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": "Bearer"},
        {"Authorization": "bearer " + "x" * 9000},
        {"Authorization": f"Bearer {tokens.hs256(exp=1)}"},
        {"Authorization": f"Bearer {tokens.hs256(aud='someone-else')}"},
        {"Authorization": f"Bearer {tokens.hs256(iss='https://evil.example')}"},
        {"Authorization": f"Bearer {tokens.hs256(secret=OTHER_SECRET)}"},
    ],
)
def test_invalid_expired_or_foreign_tokens_are_rejected(
    stack: dict[str, Any], headers: dict[str, str]
) -> None:
    response = stack["client"].post("/api/search", json={"query": "fees"}, headers=headers)
    assert response.status_code == 401
    assert "Bearer " not in response.text


def test_a_valid_token_with_no_usable_role_is_forbidden_not_unauthorized(
    stack: dict[str, Any],
) -> None:
    response = stack["client"].post(
        "/api/search", json={"query": "fees"}, headers=bearer("janitor")
    )
    assert response.status_code == 403


def test_a_token_cannot_invent_a_role(stack: dict[str, Any]) -> None:
    body = search(stack, bearer(["superadmin", "root", "employee"]), "credit score approval")
    assert UNDERWRITING not in texts(body)


def test_me_reports_the_callers_scope(stack: dict[str, Any]) -> None:
    response = stack["client"].get(
        "/api/me", headers=bearer("underwriter", department=["Underwriting"], tenant="default")
    )
    body = response.json()
    assert body["authenticated"] is True
    assert body["roles"] == ["underwriter"]
    assert body["departments"] == ["underwriting"]
    assert body["max_access_level"] == "restricted"
    assert body["tenant_id"] == "default"


# --- role-based access -------------------------------------------------------------------------


def test_an_employee_never_receives_restricted_text_in_any_mode(stack: dict[str, Any]) -> None:
    for mode in ("semantic", "lexical", "hybrid"):
        for query in ("credit score automated approval", "retention periods records", "wire fees"):
            body = search(stack, bearer("employee"), query, mode=mode)
            assert UNDERWRITING not in texts(body), (mode, query)
            assert PRIVACY not in texts(body), (mode, query)
            assert "underwriting-policy" not in documents_of(body)
            assert "data-privacy-and-retention" not in documents_of(body)
            assert LEGAL not in texts(body)


def test_an_employee_still_finds_what_they_are_allowed_to_read(stack: dict[str, Any]) -> None:
    body = search(stack, bearer("employee"), "late payment fee grace period", mode="lexical")
    assert "fee-schedule" in documents_of(body)


def test_an_underwriter_reads_underwriting_but_not_other_departments_restricted_documents(
    stack: dict[str, Any],
) -> None:
    body = search(
        stack,
        bearer("underwriter", department=["underwriting"]),
        "automated approval",
        mode="lexical",
    )
    assert UNDERWRITING in texts(body)
    privacy = search(
        stack,
        bearer("underwriter", department=["underwriting"]),
        "records kept years",
        mode="lexical",
    )
    assert PRIVACY not in texts(privacy)  # compliance's restricted document


def test_a_restricted_role_without_the_department_sees_no_restricted_documents(
    stack: dict[str, Any],
) -> None:
    body = search(stack, bearer("underwriter"), "automated approval", mode="lexical")
    assert UNDERWRITING not in texts(body)


def test_compliance_reads_the_privacy_policy_but_not_underwriting(stack: dict[str, Any]) -> None:
    headers = bearer("compliance", department=["compliance"])
    assert PRIVACY in texts(search(stack, headers, "records kept years", mode="lexical"))
    assert UNDERWRITING not in texts(search(stack, headers, "automated approval", mode="lexical"))


def test_confidential_documents_are_for_admins_only(stack: dict[str, Any]) -> None:
    for role, department in (
        ("employee", None),
        ("underwriter", ["legal"]),
        ("compliance", ["legal"]),
    ):
        assert LEGAL not in texts(
            search(stack, bearer(role, department=department), "litigation hold", mode="lexical")
        )
    assert LEGAL in texts(search(stack, bearer("admin"), "litigation hold", mode="lexical"))


def test_an_admin_reads_every_department_but_only_in_their_own_tenant(
    stack: dict[str, Any],
) -> None:
    admin = bearer("admin")
    everything = texts(
        search(stack, admin, "automated approval records litigation", mode="lexical")
    )
    assert UNDERWRITING in everything
    assert LEGAL in everything
    assert ACME not in everything


# --- tenant isolation and spoofing -------------------------------------------------------------


def test_tenants_are_isolated_in_both_directions(stack: dict[str, Any]) -> None:
    acme = search(stack, bearer("admin", tenant="acme"), "automated approval", mode="lexical")
    assert ACME in texts(acme)
    assert documents_of(acme) == {"acme-underwriting"}  # none of the default tenant's documents
    default = search(stack, bearer("admin", tenant="default"), "automated approval", mode="lexical")
    assert ACME not in texts(default)


def test_a_caller_cannot_choose_a_tenant_through_the_request(stack: dict[str, Any]) -> None:
    headers = {
        **bearer("admin", tenant="default"),
        "X-Tenant-ID": "acme",
        "X-Forwarded-User": "acme-admin",
    }
    response = stack["client"].post(
        "/api/search?tenant_id=acme",
        json={"query": "automated approval", "mode": "lexical"},
        headers=headers,
    )
    assert ACME not in response.text
    # ...and tenant_id is not an accepted filter, so it cannot be smuggled in the body either.
    bad = stack["client"].post(
        "/api/search",
        json={"query": "x", "filters": {"tenant_id": "acme"}},
        headers=bearer("admin"),
    )
    assert bad.status_code == 422
    bad = stack["client"].post(
        "/api/search", json={"query": "x", "tenant_id": "acme"}, headers=bearer("admin")
    )
    assert bad.status_code == 422


def test_a_hostile_tenant_claim_is_rejected_before_it_reaches_sql(stack: dict[str, Any]) -> None:
    headers = bearer("admin", tenant="default' OR '1'='1")
    response = stack["client"].post("/api/search", json={"query": "x"}, headers=headers)
    assert response.status_code == 401


# --- caller filters cannot widen access --------------------------------------------------------


def test_caller_filters_can_narrow_but_never_widen_the_scope(stack: dict[str, Any]) -> None:
    employee = bearer("employee")
    for filters in (
        {"category": "underwriting"},
        {"document": "underwriting-policy"},
        {"status": "current", "category": "underwriting"},
        {"version": "2.0", "document": "underwriting-policy"},
        {"section": "Credit score decision bands"},
    ):
        body = search(
            stack, employee, "credit score automated approval", filters=filters, mode="hybrid"
        )
        assert body["results"] == [], filters


def test_the_documents_listing_only_names_documents_the_caller_may_read(
    stack: dict[str, Any],
) -> None:
    def names(headers: dict[str, str]) -> set[str]:
        response = stack["client"].get("/api/documents", headers=headers)
        assert response.status_code == 200
        return {d["name"] for d in response.json()}

    employee = names(bearer("employee"))
    assert "underwriting-policy" not in employee
    assert "data-privacy-and-retention" not in employee
    assert "legal-hold" not in employee
    assert {"fee-schedule", "complaint-handling"} <= employee
    assert "underwriting-policy" in names(bearer("underwriter", department=["underwriting"]))
    assert "legal-hold" in names(bearer("admin"))
    assert "acme-underwriting" not in names(bearer("admin"))
    assert names(bearer("admin", tenant="acme")) == {"acme-underwriting"}


# --- the generator, the reranker and the observability pipeline never see restricted text ------


def test_restricted_chunks_never_reach_the_reranker(stack: dict[str, Any]) -> None:
    reranker: KeywordReranker = stack["reranker"]
    search(stack, bearer("employee"), "credit score automated approval", rerank=True, mode="hybrid")
    seen = " ".join(p for _, passages in reranker.calls for p in passages)
    assert seen  # authorized candidates were reranked
    assert UNDERWRITING not in seen
    assert PRIVACY not in seen
    assert LEGAL not in seen


def test_unauthorized_content_never_reaches_the_generator(stack: dict[str, Any]) -> None:
    generator: SpyGenerator = stack["generator"]
    client: TestClient = stack["client"]
    question = "What credit score gets an applicant approved automatically?"

    employee = client.post(
        "/api/ask", json={"question": question}, headers=bearer("employee")
    ).json()
    assert UNDERWRITING not in " ".join(generator.seen)
    assert all(s["document"] != "underwriting-policy" for s in employee["sources"])
    assert UNDERWRITING not in json.dumps(employee)

    generator.seen.clear()  # positive control: the same question, with clearance
    underwriter = client.post(
        "/api/ask",
        json={"question": question},
        headers=bearer("underwriter", department=["underwriting"]),
    ).json()
    assert UNDERWRITING in " ".join(generator.seen)
    assert any(s["document"] == "underwriting-policy" for s in underwriter["sources"])


def test_restricted_text_never_appears_in_traces_or_logs(stack: dict[str, Any]) -> None:
    client: TestClient = stack["client"]
    client.post(
        "/api/ask", json={"question": "automated approval credit score"}, headers=bearer("employee")
    )
    search(stack, bearer("employee"), "automated approval", mode="hybrid", rerank=True)
    everything = json.dumps([s.to_json() for s in stack["exporter"].get_finished_spans()])
    everything += "\n".join(stack["logs"].lines)
    for marker in (UNDERWRITING, PRIVACY, ACME, LEGAL):
        assert marker not in everything


def test_auth_failures_are_logged_without_the_token(stack: dict[str, Any]) -> None:
    token = tokens.hs256(exp=1)
    stack["client"].post(
        "/api/search", json={"query": "x"}, headers={"Authorization": f"Bearer {token}"}
    )
    lines = [json.loads(line) for line in stack["logs"].lines]
    failed = [e for e in lines if e["event"] == "auth.failed"]
    assert failed
    assert failed[0]["reason"] == "expired"
    assert token not in "\n".join(stack["logs"].lines)


# --- fail closed -------------------------------------------------------------------------------


def test_chunks_without_authorization_labels_are_invisible_to_everyone(
    stack: dict[str, Any],
) -> None:
    seed_chunks(stack["url"], [Spec("UNLABELLEDMARKER legacy chunk", basis(0), document="legacy")])
    with create_pool(
        stack["url"]
    ).connection() as conn:  # strip the labels, as a legacy row would be
        conn.execute(
            "UPDATE chunks SET metadata = metadata - 'tenant_id' - 'access_rank'"
            " - 'department' - 'access_level' WHERE chunk_text LIKE 'UNLABELLEDMARKER%'"
        )
    for role in ("employee", "admin"):
        assert "UNLABELLEDMARKER" not in texts(
            search(stack, bearer(role), "UNLABELLEDMARKER legacy", mode="lexical")
        )


# --- the MCP interface obeys the same boundary -------------------------------------------------


def mcp_call(
    stack: dict[str, Any], scope_headers: dict[str, str], name: str, arguments: dict[str, Any]
) -> Any:
    token = scope_headers["Authorization"].split(" ", 1)[1]
    settings = app_settings(stack["url"])
    scope = TokenValidator(settings).scope(token)
    embedder = HashingEmbedder(TEST_DIM)
    services = open_services(settings, embedder, reranker=KeywordReranker())
    try:
        server = build_server(
            PolicyTools(services.retrieval, services.rag, services.pool, access=scope)
        )

        async def go() -> Any:
            async with Client(server) as client:
                return await client.call_tool(name, arguments)

        return anyio.run(go)
    finally:
        services.close()


def test_mcp_search_and_documents_respect_the_scope(stack: dict[str, Any]) -> None:
    employee = bearer("employee")
    found = mcp_call(
        stack,
        employee,
        "search_policy",
        {"query": "automated approval", "top_k": 10, "mode": "lexical"},
    )
    assert UNDERWRITING not in json.dumps(found.structured_content)
    blocked = mcp_call(stack, employee, "get_policy_document", {"document": "underwriting-policy"})
    assert blocked.is_error
    allowed = mcp_call(stack, employee, "get_policy_document", {"document": "fee-schedule"})
    assert not allowed.is_error
    privileged = mcp_call(
        stack,
        bearer("underwriter", department=["underwriting"]),
        "get_policy_document",
        {"document": "underwriting-policy"},
    )
    assert not privileged.is_error


def test_mcp_ask_never_hands_restricted_text_to_the_generator(stack: dict[str, Any]) -> None:
    result = mcp_call(
        stack,
        bearer("employee"),
        "ask_policy",
        {"question": "credit score for automated approval?"},
    )
    assert UNDERWRITING not in json.dumps(result.structured_content)


# --- configuration and demo mode ---------------------------------------------------------------


def test_misconfigured_authentication_stops_the_app_instead_of_running_open() -> None:
    with pytest.raises(ValueError, match="exactly one key source"):
        create_app(tokens.settings(auth_jwt_secret=None))
    with pytest.raises(ValueError, match="32 characters"):
        create_app(tokens.settings(auth_jwt_secret="short"))


def test_demo_mode_is_explicit_and_unrestricted(clean_db: str) -> None:
    ingest_all(clean_db)
    settings = Settings(database_url=clean_db, embedding_dim=TEST_DIM, _env_file=None)  # type: ignore[call-arg]
    assert settings.auth_mode == "off"  # the default, documented as a local demo mode
    app = create_app(settings, HashingEmbedder(TEST_DIM))
    try:
        with TestClient(app) as client:
            me = client.get("/api/me").json()
            body = client.post(
                "/api/search", json={"query": "automated approval", "mode": "lexical", "top_k": 20}
            ).json()
    finally:
        set_current(Telemetry())
    assert me["authenticated"] is False
    assert me["auth_mode"] == "off"
    assert UNDERWRITING in " ".join(h["text"] for h in body["results"])


# --- a real MCP client over stdio, with the real model ------------------------------------------


@pytest.mark.model
def test_the_stdio_mcp_server_enforces_authentication_and_scope(clean_db: str) -> None:
    settings = app_settings(clean_db)
    ingest_all(clean_db, FastEmbedProvider(settings.embedding_model))
    base_env = {
        **os.environ,
        "DATABASE_URL": clean_db,
        "AUTO_INIT_SCHEMA": "false",
        "AUTH_MODE": "jwt",
        "AUTH_ISSUER": tokens.ISSUER,
        "AUTH_AUDIENCE": tokens.AUDIENCE,
        "AUTH_JWT_SECRET": tokens.SECRET,
    }

    # No credential: the server refuses to start rather than run open.
    refused = subprocess.run(
        [sys.executable, "-m", "app.mcp_server"],
        env={**base_env, "MCP_ACCESS_TOKEN": ""},
        capture_output=True,
        text=True,
        timeout=60,
        input="",
    )
    assert refused.returncode == 2
    assert "not started" in refused.stderr

    bad_token = tokens.hs256(exp=1)
    expired = subprocess.run(
        [sys.executable, "-m", "app.mcp_server"],
        env={**base_env, "MCP_ACCESS_TOKEN": bad_token},
        capture_output=True,
        text=True,
        timeout=60,
        input="",
    )
    assert expired.returncode == 2
    assert bad_token not in expired.stderr  # the token is never echoed

    # A valid employee token: the server starts, and restricted documents are out of reach.
    employee_token = tokens.hs256(roles=["employee"], tenant_id="default")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        env={**base_env, "MCP_ACCESS_TOKEN": employee_token},
        cwd=Path.cwd(),
    )

    async def go() -> tuple[Any, Any, Any]:
        async with Client(params) as client:
            found = await client.call_tool(
                "search_policy",
                {"query": "What credit score is eligible for automated approval?", "top_k": 10},
            )
            restricted = await client.call_tool(
                "get_policy_document", {"document": "underwriting-policy"}
            )
            open_doc = await client.call_tool("get_policy_document", {"document": "fee-schedule"})
            return found, restricted, open_doc

    found, restricted, open_doc = anyio.run(go)
    passages = " ".join(h["text"] for h in (found.structured_content or {})["results"])
    assert passages  # the employee still gets answers from documents they may read
    assert UNDERWRITING not in passages
    assert restricted.is_error
    assert not open_doc.is_error
