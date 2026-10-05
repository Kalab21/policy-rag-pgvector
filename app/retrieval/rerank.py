"""Cross-encoder reranking of a small candidate set.

A bi-encoder (the embedding model) maps the query and each passage to vectors independently,
which makes searching the whole corpus cheap. A cross-encoder reads the query and one passage
TOGETHER and scores how well the passage answers the query: more accurate, but far too slow
to run on every chunk, so it only re-orders the few candidates the retrievers already found.
"""

import os
from dataclasses import replace
from typing import Any, Protocol

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.models.domain import RetrievedChunk

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    @property
    def model_name(self) -> str: ...

    def score(self, query: str, passages: list[str]) -> list[float]:
        """One relevance score per passage, higher meaning more relevant."""
        ...


class CrossEncoderReranker:
    """A local ONNX cross-encoder served by fastembed (no API key, no network after download)."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        cache_dir: str | None = None,
        model: Any = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir or os.environ.get("FASTEMBED_CACHE_DIR")
        self._model = model  # injectable for tests; otherwise loaded on first use

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        if self._model is None:
            self._model = TextCrossEncoder(model_name=self._model_name, cache_dir=self._cache_dir)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = [float(s) for s in self._load().rerank(query, passages)]
        if len(scores) != len(passages):
            raise RuntimeError("the reranker returned the wrong number of scores")
        return scores


def passage_text(chunk: RetrievedChunk) -> str:
    """What the cross-encoder reads: the title and section give it the same context the
    embedding model had, followed by the chunk text."""
    return f"{chunk.title} | {chunk.section}\n{chunk.text}"


def rerank_chunks(
    reranker: Reranker, query: str, chunks: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    """Re-order `chunks` by cross-encoder score and keep the best `top_k`.

    Ties are broken by chunk id so the result is deterministic. Each returned chunk keeps its
    cosine similarity (the evidence gate depends on it) and gains its `rerank_score`.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not chunks:
        return []
    scores = reranker.score(query, [passage_text(c) for c in chunks])
    ranked = sorted(zip(scores, chunks, strict=True), key=lambda pair: (-pair[0], pair[1].chunk_id))
    return [replace(c, rerank_score=s) for s, c in ranked[:top_k]]
