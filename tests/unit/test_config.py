import pytest
from pydantic import ValidationError

from app.core.config import MAX_INDEXABLE_DIM, Settings


def test_defaults_describe_the_local_embedding_model() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding_dim == 384
    assert settings.auto_init_schema is True


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/x")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.embedding_dim == 768
    assert settings.database_url == "postgresql://u:p@db:5432/x"


@pytest.mark.parametrize("dim", [0, -1, MAX_INDEXABLE_DIM + 1])
def test_rejects_dimensions_pgvector_cannot_index(dim: int) -> None:
    with pytest.raises(ValidationError):
        Settings(embedding_dim=dim, _env_file=None)  # type: ignore[call-arg]
