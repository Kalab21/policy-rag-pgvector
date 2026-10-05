"""Retrieval service: embed the query with the same model as the documents, then search."""

import time
from collections.abc import Mapping
from typing import Literal, get_args

from app.db.pool import DictPool
from app.embeddings.base import EmbeddingProvider
from app.models.domain import RetrievedChunk
from app.observability.telemetry import current, set_attributes, span
from app.retrieval.fusion import DEFAULT_RRF_K
from app.retrieval.rerank import Reranker, rerank_chunks
from app.retrieval.store import search_chunks, search_hybrid, search_lexical
from app.security.access import AccessScope

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
        reranker: Reranker | None = None,
        rerank_enabled: bool = False,
        rerank_candidates: int = 20,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self._ef_search = ef_search
        self._iterative_scan = iterative_scan
        self._mode = mode
        self._rrf_k = rrf_k
        self._candidates = candidates
        self._reranker = reranker
        self._rerank_enabled = rerank_enabled
        self._rerank_candidates = rerank_candidates

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    @property
    def rerank_enabled(self) -> bool:
        return self._rerank_enabled

    @property
    def mode(self) -> RetrievalMode:
        return self._mode

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Mapping[str, str] | None = None,
        mode: RetrievalMode | None = None,
        rerank: bool | None = None,
        access: AccessScope | None = None,
    ) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        chosen = mode or self._mode
        if chosen not in RETRIEVAL_MODES:
            raise ValueError(f"unknown retrieval mode {chosen!r}; use one of {RETRIEVAL_MODES}")
        use_rerank = self._rerank_enabled if rerank is None else rerank
        if use_rerank and self._reranker is None:
            raise ValueError("reranking was requested but no reranker is configured")
        # With reranking, the retriever hands over a candidate set (never the whole corpus)
        # and the cross-encoder picks the final top_k from it.
        depth = max(self._rerank_candidates, top_k) if use_rerank else top_k
        # The query is always embedded: even lexical results report a cosine similarity,
        # which the evidence gate relies on.
        telemetry = current()
        with span(
            "embedding.query",
            {"model": self._embedder.model_name, **telemetry.query_attributes(query)},
        ):
            vector = self._embedder.embed_query(query)
        started = time.perf_counter()
        found = self._retrieve(chosen, query, vector, depth, filters, access)
        telemetry.record("retrieval_ms", (time.perf_counter() - started) * 1000, {"mode": chosen})
        if use_rerank and self._reranker is not None:
            reranker = self._reranker
            started = time.perf_counter()
            with span(
                "retrieval.rerank",
                {"model": reranker.model_name, "candidates": len(found), "top_k": top_k},
            ) as sp:
                found = rerank_chunks(reranker, query, found, top_k)
                set_attributes(sp, {"returned": len(found)})
            telemetry.record("rerank_ms", (time.perf_counter() - started) * 1000)
        telemetry.record("returned_chunks", len(found), {"mode": chosen})
        return found

    def _retrieve(
        self,
        mode: str,
        query: str,
        vector: list[float],
        depth: int,
        filters: Mapping[str, str] | None,
        access: AccessScope | None,
    ) -> list[RetrievedChunk]:
        if mode == "semantic":
            return search_chunks(
                self._pool, vector, depth, filters, self._ef_search, self._iterative_scan, access
            )
        if mode == "lexical":
            return search_lexical(self._pool, query, vector, depth, filters, access)
        return search_hybrid(
            self._pool,
            query,
            vector,
            depth,
            filters,
            candidates=max(self._candidates, depth),
            rrf_k=self._rrf_k,
            ef_search=self._ef_search,
            iterative_scan=self._iterative_scan,
            access=access,
        )
