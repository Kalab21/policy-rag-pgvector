"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.embeddings.base import EmbeddingProvider
from app.observability.logs import configure_logging
from app.observability.setup import install_http_observability, telemetry_from_settings
from app.observability.telemetry import Telemetry, set_current
from app.rag.generator import AnswerGenerator
from app.retrieval.rerank import Reranker
from app.services import open_services


def create_app(
    settings: Settings | None = None,
    embedder: EmbeddingProvider | None = None,
    generator: AnswerGenerator | None = None,
    reranker: Reranker | None = None,
    telemetry: Telemetry | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_format, settings.log_level)
    telemetry = telemetry or telemetry_from_settings(settings)
    set_current(telemetry)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        services = open_services(settings, embedder, generator, reranker)
        app.state.pool = services.pool
        app.state.settings = settings
        app.state.retrieval = services.retrieval
        app.state.rag = services.rag
        app.state.telemetry = telemetry
        try:
            yield
        finally:
            services.close()
            telemetry.shutdown()

    app = FastAPI(
        title="Policy RAG Platform",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    install_http_observability(app, telemetry)
    return app


app = create_app()
