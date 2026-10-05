"""Retrieval service: embed the query with the same model as the documents, then search."""

from collections.abc import Mapping
from typing import Literal, get_args

from app.db.pool import DictPool
from app.embeddings.base import EmbeddingProvider
from app.models.domain import RetrievedChunk
from app.retrieval.fusion import DEFAULT_RRF_K
from app.retrieval.store import search_chunks, search_hybrid, search_lexical

RetrievalMode = Literal["semantic", "lexical", "hybrid"]
RETRIEVAL_MODES: tuple[str, ...] = get_args(RetrievalMode)


class RetrievalService:
    def __init__(
        self,
        pool: DictPool,
        embedder: EmbeddingProvider,
        ef_search: int = 40,
        iterative_scan: bool = False,
        mode: RetrievalMode = "semantic",
        rrf_k: int = DEFAULT_RRF_K,
        candidates: int = 30,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self._ef_search = ef_search
        self._iterative_scan = iterative_scan
        self._mode = mode
        self._rrf_k = rrf_k
        self._candidates = candidates

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    @property
    def mode(self) -> RetrievalMode:
        return self._mode

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Mapping[str, str] | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        chosen = mode or self._mode
        if chosen not in RETRIEVAL_MODES:
            raise ValueError(f"unknown retrieval mode {chosen!r}; use one of {RETRIEVAL_MODES}")
        # The query is always embedded: even lexical results report a cosine similarity,
        # which the evidence gate relies on.
        vector = self._embedder.embed_query(query)
        if chosen == "semantic":
            return search_chunks(
                self._pool, vector, top_k, filters, self._ef_search, self._iterative_scan
            )
        if chosen == "lexical":
            return search_lexical(self._pool, query, vector, top_k, filters)
        return search_hybrid(
            self._pool,
            query,
            vector,
            top_k,
            filters,
            candidates=self._candidates,
            rrf_k=self._rrf_k,
            ef_search=self._ef_search,
            iterative_scan=self._iterative_scan,
        )
