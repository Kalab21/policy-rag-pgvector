"""Choose the embedding provider from settings."""

from app.core.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.fastembed_provider import FastEmbedProvider


class UnknownProviderError(ValueError):
    pass


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Only the local fastembed provider is implemented.

    A hosted provider would be added here by implementing `EmbeddingProvider`; nothing in
    ingestion or retrieval would change.
    """
    if settings.embedding_provider == "fastembed":
        provider = FastEmbedProvider(settings.embedding_model)
    else:
        raise UnknownProviderError(
            f"unknown EMBEDDING_PROVIDER {settings.embedding_provider!r}; implemented: fastembed"
        )
    if provider.dimension != settings.embedding_dim:
        raise ValueError(
            f"{provider.model_name} produces {provider.dimension}-dimensional vectors but "
            f"EMBEDDING_DIM is {settings.embedding_dim}"
        )
    return provider
