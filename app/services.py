"""Build the long-lived services (database pool, retrieval, RAG) from settings.

Both the HTTP API and the MCP server call this, so neither duplicates retrieval or RAG logic.
"""

from dataclasses import dataclass

from app.core.config import Settings
from app.db.pool import DictPool, create_pool
from app.db.schema import embedding_column_dim, init_schema, pgvector_version
from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import get_embedding_provider
from app.rag.factory import get_generator
from app.rag.generator import AnswerGenerator
from app.rag.service import RagService
from app.retrieval.rerank import CrossEncoderReranker, Reranker
from app.retrieval.service import RetrievalService
from app.retrieval.store import supports_iterative_scan


@dataclass
class Services:
    settings: Settings
    pool: DictPool
    retrieval: RetrievalService
    rag: RagService

    def close(self) -> None:
        self.pool.close()


def open_services(
    settings: Settings,
    embedder: EmbeddingProvider | None = None,
    generator: AnswerGenerator | None = None,
    reranker: Reranker | None = None,
) -> Services:
    if settings.auto_init_schema:
        init_schema(
            settings.database_url,
            settings.embedding_dim,
            settings.hnsw_m,
            settings.hnsw_ef_construction,
        )
    pool = create_pool(settings.database_url, settings.db_pool_min_size, settings.db_pool_max_size)
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
    retrieval = RetrievalService(
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
    rag = RagService(
        retrieval,
        generator or get_generator(settings, provider),
        settings.evidence_min_similarity,
        settings.rag_max_context_chunks,
    )
    return Services(settings=settings, pool=pool, retrieval=retrieval, rag=rag)
