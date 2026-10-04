"""Ranking metrics over retrieved items, where each item is a hashable key."""

from collections.abc import Hashable, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


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


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
