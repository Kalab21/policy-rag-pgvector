"""Ranking metrics over retrieved items, where each item is a hashable key."""

import math
from collections.abc import Hashable, Sequence


def hit_at_k[T: Hashable](ranked: Sequence[T], relevant: set[T], k: int) -> float:
    """1.0 if any relevant item is in the top `k`, else 0.0."""
    return 1.0 if any(item in relevant for item in ranked[:k]) else 0.0


def recall_at_k[T: Hashable](ranked: Sequence[T], relevant: set[T], k: int) -> float:
    """Share of the relevant items that appear in the top `k`."""
    if not relevant:
        raise ValueError("recall needs at least one relevant item")
    return len(relevant & set(ranked[:k])) / len(relevant)


def reciprocal_rank[T: Hashable](ranked: Sequence[T], relevant: set[T]) -> float:
    """1 / rank of the first relevant item (0.0 when none was retrieved)."""
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k[T: Hashable](ranked: Sequence[T], relevant: set[T], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    DCG adds 1 / log2(rank + 1) for each relevant item in the top `k`; it is divided by the
    DCG of the best possible ordering, so 1.0 means every relevant item is ranked first.
    """
    if not relevant:
        raise ValueError("nDCG needs at least one relevant item")
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0-100) by linear interpolation; 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
