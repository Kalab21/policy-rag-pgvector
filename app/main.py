"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.embeddings.base import EmbeddingProvider
from app.rag.generator import AnswerGenerator
from app.retrieval.rerank import Reranker
from app.services import open_services


def create_app(
    settings: Settings | None = None,
    embedder: EmbeddingProvider | None = None,
    generator: AnswerGenerator | None = None,
    reranker: Reranker | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        services = open_services(settings, embedder, generator, reranker)
        app.state.pool = services.pool
        app.state.settings = settings
        app.state.retrieval = services.retrieval
        app.state.rag = services.rag
        try:
            yield
        finally:
            services.close()

    app = FastAPI(
        title="Policy RAG Platform",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
