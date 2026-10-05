"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.db.pool import create_pool
from app.db.schema import embedding_column_dim, init_schema, pgvector_version
from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import get_embedding_provider
from app.rag.factory import get_generator
from app.rag.generator import AnswerGenerator
from app.rag.service import RagService
from app.retrieval.rerank import CrossEncoderReranker, Reranker
from app.retrieval.service import RetrievalService
from app.retrieval.store import supports_iterative_scan


def create_app(
    settings: Settings | None = None,
    embedder: EmbeddingProvider | None = None,
    generator: AnswerGenerator | None = None,
    reranker: Reranker | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.auto_init_schema:
            init_schema(
                settings.database_url,
                settings.embedding_dim,
                settings.hnsw_m,
                settings.hnsw_ef_construction,
            )
        pool = create_pool(
            settings.database_url, settings.db_pool_min_size, settings.db_pool_max_size
        )
        with pool.connection() as conn:
            actual = embedding_column_dim(conn)
            version = pgvector_version(conn)
        if actual != settings.embedding_dim:
            pool.close()
            raise RuntimeError(
                f"chunks.embedding has {actual} dimensions but EMBEDDING_DIM is "
                f"{settings.embedding_dim}; use the matching model or recreate the table"
            )
        provider = embedder or get_embedding_provider(settings)
        app.state.pool = pool
        app.state.settings = settings
        app.state.retrieval = RetrievalService(
            pool,
            provider,
            ef_search=settings.hnsw_ef_search,
            iterative_scan=supports_iterative_scan(version),
            mode=settings.retrieval_mode,
            rrf_k=settings.rrf_k,
            candidates=settings.hybrid_candidates,
            reranker=reranker or CrossEncoderReranker(settings.rerank_model),
            rerank_enabled=settings.rerank_enabled,
            rerank_candidates=settings.rerank_candidates,
        )
        app.state.rag = RagService(
            app.state.retrieval,
            generator or get_generator(settings, provider),
            settings.evidence_min_similarity,
            settings.rag_max_context_chunks,
        )
        try:
            yield
        finally:
            pool.close()

    app = FastAPI(
        title="Policy RAG Platform",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
