"""A deterministic stand-in for the embedding model.

It maps each word to a fixed bucket, so texts that share words get similar vectors. That is
enough to exercise ingestion, storage, filtering and ranking mechanics quickly, with the
real pgvector doing the storage and the similarity search. Tests that depend on semantic
quality use the real model instead (marked `model`).
"""

import hashlib
import math
import re


class HashingEmbedder:
    model_name = "test-hashing-embedder"

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self.calls = 0  # number of embed_documents calls, to prove re-embedding is skipped

    @property
    def dimension(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self._dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            values[digest % self._dim] += 1.0 if (digest >> 100) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FailingEmbedder(HashingEmbedder):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding service unavailable")


class KeywordReranker:
    """A deterministic stand-in for the cross-encoder: scores a passage by how many of the
    query's words it contains. Records what it was asked to score."""

    model_name = "test-keyword-reranker"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        words = set(query.lower().split())
        return [float(len(words & set(p.lower().replace("\n", " ").split()))) for p in passages]


class FakeBedrockClient:
    """Stands in for boto3's bedrock-runtime client (the one true external boundary)."""

    def __init__(self, response: dict | None = None, error: Exception | None = None) -> None:  # type: ignore[type-arg]
        self.response = response
        self.error = error
        self.requests: list[dict] = []  # type: ignore[type-arg]

    def converse(self, **kwargs):  # type: ignore[no-untyped-def]
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def tool_response(payload: object, name: str = "submit_answer") -> dict:  # type: ignore[type-arg]
    """A Converse response in which the model called a tool with `payload`."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "t1", "name": name, "input": payload}}],
            }
        },
        "stopReason": "tool_use",
    }
