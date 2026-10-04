"""Embedding provider: the factory logic, and the real model (marked `model`)."""

import math

import pytest

from app.core.config import Settings
from app.embeddings.factory import UnknownProviderError, get_embedding_provider
from app.embeddings.fastembed_provider import FastEmbedProvider


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_the_default_model_reports_384_dimensions_without_loading() -> None:
    provider = get_embedding_provider(_settings())
    assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert provider.dimension == 384


def test_an_unknown_provider_is_rejected() -> None:
    with pytest.raises(UnknownProviderError, match="implemented: fastembed"):
        get_embedding_provider(_settings(embedding_provider="openai"))


def test_a_mismatch_between_model_and_configured_width_is_rejected() -> None:
    with pytest.raises(ValueError, match="384-dimensional"):
        get_embedding_provider(_settings(embedding_dim=768))


def test_an_unsupported_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not support"):
        _ = FastEmbedProvider("not/a-real-model").dimension


@pytest.fixture(scope="module")
def real_provider() -> FastEmbedProvider:
    return FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2")


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.mark.model
def test_real_vectors_have_the_declared_width_and_unit_length(
    real_provider: FastEmbedProvider,
) -> None:
    vectors = real_provider.embed_documents(["Late fees are capped.", "Another passage."])
    assert len(vectors) == 2
    assert all(len(v) == real_provider.dimension == 384 for v in vectors)
    assert all(math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, abs_tol=1e-6) for v in vectors)


@pytest.mark.model
def test_documents_and_queries_use_the_same_vector_space(real_provider: FastEmbedProvider) -> None:
    text = "The late payment fee is the lesser of $35 or 5 percent of the installment."
    document_vector = real_provider.embed_documents([text])[0]
    query_vector = real_provider.embed_query(text)
    assert _cosine(document_vector, query_vector) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.model
def test_semantically_related_text_scores_higher_than_unrelated_text(
    real_provider: FastEmbedProvider,
) -> None:
    query = real_provider.embed_query("What is the late payment fee?")
    related = real_provider.embed_documents(["A late fee is charged when a payment is overdue."])[0]
    unrelated = real_provider.embed_documents(["The office cafeteria serves lunch at noon."])[0]
    assert _cosine(query, related) > _cosine(query, unrelated) + 0.2


@pytest.mark.model
def test_embedding_the_same_text_twice_gives_the_same_vector(
    real_provider: FastEmbedProvider,
) -> None:
    first = real_provider.embed_query("debt-to-income limit")
    second = real_provider.embed_query("debt-to-income limit")
    assert _cosine(first, second) == pytest.approx(1.0, abs=1e-6)
