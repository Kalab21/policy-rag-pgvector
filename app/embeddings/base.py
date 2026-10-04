"""The embedding-provider interface.

Ingestion and retrieval depend only on this protocol, so a different provider (a hosted
API, for example) can be added without changing either of them. Documents and queries
must be embedded with the same model, which is why one object provides both operations.
"""

from typing import Protocol


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str:
        """Identifier recorded with every stored document."""
        ...

    @property
    def dimension(self) -> int:
        """Width of the vectors this provider returns."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for storage."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        ...
