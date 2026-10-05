"""Reciprocal Rank Fusion: merge several ranked lists into one."""

from collections.abc import Mapping, Sequence
from dataclasses import replace

from app.models.domain import RetrievedChunk

DEFAULT_RRF_K = 60  # the constant from the original RRF paper; damps the weight of top ranks


def rrf_fuse(
    rankings: Mapping[str, Sequence[RetrievedChunk]],
    k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """Fuse ranked lists with RRF: score(chunk) = sum over lists of 1 / (k + rank).

    `rankings` maps a retriever name to its results, best first. Only ranks are used, so
    scores from different retrievers (cosine similarity, text rank) never need comparing.
    The result is ordered by fused score, with ties broken by chunk id so the order is
    deterministic. Each result carries its fused `score` and the retrievers that found it.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    scores: dict[int, float] = {}
    found_by: dict[int, list[str]] = {}
    first_seen: dict[int, RetrievedChunk] = {}
    for name, ranked in rankings.items():
        seen: set[int] = set()
        for rank, chunk in enumerate(ranked, start=1):
            if chunk.chunk_id in seen:  # a retriever listing a chunk twice counts once
                continue
            seen.add(chunk.chunk_id)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            found_by.setdefault(chunk.chunk_id, []).append(name)
            first_seen.setdefault(chunk.chunk_id, chunk)
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [
        replace(first_seen[cid], score=scores[cid], matched_by=tuple(found_by[cid]))
        for cid in order
    ]
