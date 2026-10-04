"""Local embeddings with fastembed (ONNX runtime), so no API key or network call is needed
after the model has been downloaded once.

The default model is `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, Apache-2.0).
fastembed serves an ONNX export of it. The model truncates input at 256 word pieces, which
the default chunk size stays well below.
"""

import math
import os
from typing import Any

from fastembed import TextEmbedding


class FastEmbedProvider:
    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir or os.environ.get("FASTEMBED_CACHE_DIR")
        self._model: Any = None
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            for info in TextEmbedding.list_supported_models():
                if info["model"] == self._model_name:
                    self._dimension = int(info["dim"])
                    break
            else:
                raise ValueError(f"fastembed does not support the model {self._model_name!r}")
        return self._dimension

    def _load(self) -> Any:
        if self._model is None:
            self._model = TextEmbedding(model_name=self._model_name, cache_dir=self._cache_dir)
        return self._model

    @staticmethod
    def _unit(vector: Any) -> list[float]:
        values = [float(x) for x in vector]
        norm = math.sqrt(sum(x * x for x in values))
        return [x / norm for x in values] if norm else values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._unit(v) for v in self._load().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._unit(next(iter(self._load().embed([text]))))
