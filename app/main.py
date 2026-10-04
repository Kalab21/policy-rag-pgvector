"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.db.pool import create_pool
from app.db.schema import embedding_column_dim, init_schema


def create_app(settings: Settings | None = None) -> FastAPI:
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
        if actual != settings.embedding_dim:
            pool.close()
            raise RuntimeError(
                f"chunks.embedding has {actual} dimensions but EMBEDDING_DIM is "
                f"{settings.embedding_dim}; use the matching model or recreate the table"
            )
        app.state.pool = pool
        app.state.settings = settings
        try:
            yield
        finally:
            pool.close()

    app = FastAPI(
        title="Policy RAG — pgvector Retrieval Platform",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
