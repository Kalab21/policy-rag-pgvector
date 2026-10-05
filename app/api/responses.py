"""Turn domain results into API response models. Shared by the HTTP routes and the MCP tools,
so both interfaces return exactly the same shapes."""

from collections.abc import Mapping, Sequence

from app.models.domain import RetrievedChunk
from app.models.schemas import AskResponse, Citation, EvidenceInfo, SearchHit, SearchResponse
from app.rag.service import AskResult


def search_hit(h: RetrievedChunk) -> SearchHit:
    return SearchHit(
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


def search_response(
    *,
    query: str,
    top_k: int,
    filters: Mapping[str, str],
    mode: str,
    rerank: bool,
    embedding_model: str,
    hits: Sequence[RetrievedChunk],
) -> SearchResponse:
    return SearchResponse(
        query=query,
        top_k=top_k,
        filters=dict(filters),
        mode=mode,
        rerank=rerank,
        embedding_model=embedding_model,
        results=[search_hit(h) for h in hits],
    )


def ask_response(result: AskResult) -> AskResponse:
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
