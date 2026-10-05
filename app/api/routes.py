"""HTTP routes."""

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

from app.api.responses import ask_response, search_response
from app.db.schema import embedding_column_dim, pgvector_version
from app.ingestion.catalog import list_documents
from app.models.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    HealthResponse,
    MeResponse,
    SearchRequest,
    SearchResponse,
)
from app.rag.generator import GenerationError
from app.security.deps import CallerScope

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    """Liveness plus a real database check: the pgvector extension and the vector column."""
    try:
        with request.app.state.pool.connection() as conn:
            version = pgvector_version(conn)
            dim = embedding_column_dim(conn)
    except Exception as exc:  # any database failure means "not ready"
        response.status_code = 503
        return HealthResponse(status="unavailable", detail=type(exc).__name__)

    if version is None or dim is None:
        response.status_code = 503
        return HealthResponse(
            status="unavailable",
            pgvector_version=version,
            embedding_dim=dim,
            detail="pgvector extension or chunks table is missing",
        )
    return HealthResponse(status="ok", pgvector_version=version, embedding_dim=dim)


@router.get("/api/documents", response_model=list[DocumentInfo])
def documents(request: Request, scope: CallerScope) -> list[DocumentInfo]:
    """The documents that have been ingested, with their chunk counts. With authentication on,
    only the documents the caller may read."""
    return [DocumentInfo(**row) for row in list_documents(request.app.state.pool, scope)]


@router.get("/api/me", response_model=MeResponse)
def me(request: Request, scope: CallerScope) -> MeResponse:
    """What the server knows about the caller: roles, tenant, departments and the highest
    access level they may read."""
    if scope is None:
        return MeResponse(
            auth_mode=request.app.state.settings.auth_mode,
            authenticated=False,
            note="authentication is off (local demo mode): every document is readable",
        )
    return MeResponse(
        auth_mode="jwt",
        authenticated=True,
        roles=list(scope.roles),
        tenant_id=scope.tenant_id,
        departments=sorted(scope.departments),
        max_access_level=scope.max_level,
    )


@router.post("/api/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request, scope: CallerScope) -> SearchResponse:
    """Semantic search: embed the query, then return the top-k closest chunks by cosine
    distance, optionally restricted to chunks whose metadata matches `filters`."""
    service = request.app.state.retrieval
    filters = body.filters.as_dict() if body.filters else {}
    hits = service.search(body.query, body.top_k, filters, body.mode, body.rerank, scope)
    return search_response(
        query=body.query,
        top_k=body.top_k,
        filters=filters,
        mode=body.mode or service.mode,
        rerank=service.rerank_enabled if body.rerank is None else body.rerank,
        embedding_model=service.model_name,
        hits=hits,
    )


@router.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request, scope: CallerScope) -> AskResponse:
    """Answer from the policy documents, with citations, or refuse when the retrieved
    evidence is not strong enough. Answers come from current policy unless `filters.status`
    says otherwise."""
    filters = body.filters.as_dict() if body.filters else {}
    try:
        result = request.app.state.rag.ask(body.question, body.top_k, filters, scope)
    except (httpx.HTTPError, GenerationError) as exc:
        raise HTTPException(status_code=502, detail="answer generator unavailable") from exc
    return ask_response(result)
