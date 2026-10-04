"""Retrieval service: embed the query with the same model as the documents, then search."""

from collections.abc import Mapping

from app.db.pool import DictPool
from app.embeddings.base import EmbeddingProvider
from app.models.domain import RetrievedChunk
from app.retrieval.store import search_chunks


class RetrievalService:
    def __init__(
        self,
        pool: DictPool,
        embedder: EmbeddingProvider,
        ef_search: int = 40,
        iterative_scan: bool = False,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self._ef_search = ef_search
        self._iterative_scan = iterative_scan

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Mapping[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        vector = self._embedder.embed_query(query)
        return search_chunks(
            self._pool,
            vector,
            top_k,
            filters,
            ef_search=self._ef_search,
            iterative_scan=self._iterative_scan,
        )
