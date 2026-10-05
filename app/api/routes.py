"""HTTP routes."""

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

from app.db.schema import embedding_column_dim, pgvector_version
from app.ingestion.catalog import list_documents
from app.models.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    DocumentInfo,
    EvidenceInfo,
    HealthResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

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
def documents(request: Request) -> list[DocumentInfo]:
    """The documents that have been ingested, with their chunk counts."""
    return [DocumentInfo(**row) for row in list_documents(request.app.state.pool)]


@router.post("/api/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request) -> SearchResponse:
    """Semantic search: embed the query, then return the top-k closest chunks by cosine
    distance, optionally restricted to chunks whose metadata matches `filters`."""
    service = request.app.state.retrieval
    filters = body.filters.as_dict() if body.filters else {}
    hits = service.search(body.query, body.top_k, filters, body.mode, body.rerank)
    return SearchResponse(
        query=body.query,
        top_k=body.top_k,
        filters=filters,
        mode=body.mode or service.mode,
        rerank=service.rerank_enabled if body.rerank is None else body.rerank,
        embedding_model=service.model_name,
        results=[
            SearchHit(
                chunk_id=h.chunk_id,
                document=h.document,
                title=h.title,
                version=h.version,
                category=h.category,
                status=h.status,
                section=h.section,
                text=h.text,
                distance=h.distance,
                similarity=h.similarity,
                score=h.score,
                rerank_score=h.rerank_score,
                matched_by=list(h.matched_by),
                metadata=h.metadata,
            )
            for h in hits
        ],
    )


@router.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    """Answer from the policy documents, with citations, or refuse when the retrieved
    evidence is not strong enough. Answers come from current policy unless `filters.status`
    says otherwise."""
    filters = body.filters.as_dict() if body.filters else {}
    try:
        result = request.app.state.rag.ask(body.question, body.top_k, filters)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="answer generator unavailable") from exc
    return AskResponse(
        question=result.question,
        answer=result.answer,
        status=result.status,
        refusal_reason=result.refusal_reason,
        evidence=EvidenceInfo(**vars(result.evidence)),
        sources=[
            Citation(
                citation=number,
                chunk_id=c.chunk_id,
                document=c.document,
                title=c.title,
                version=c.version,
                category=c.category,
                status=c.status,
                section=c.section,
                similarity=c.similarity,
                text=c.text,
            )
            for number, c in result.sources
        ],
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        generator=result.generator,
        embedding_model=result.embedding_model,
    )
