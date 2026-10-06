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


# --- the database given in parts (as on AWS, where the password comes from Secrets Manager) ---


def parts(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "db_host": "db.internal",
        "db_name": "policy_rag",
        "db_user": "policy_rag",
        "db_password": "s3cret",
        **overrides,
    }
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def test_database_parts_build_the_connection_url() -> None:
    settings = parts()
    assert settings.database_url == (
        "postgresql://policy_rag:s3cret@db.internal:5432/policy_rag?sslmode=require"
    )


def test_special_characters_in_the_user_and_password_are_url_encoded() -> None:
    settings = parts(db_user="u@x", db_password="p@ss:/word%?#")
    assert "u%40x:p%40ss%3A%2Fword%25%3F%23@db.internal" in settings.database_url
    assert "p@ss:/word" not in settings.database_url


def test_the_port_and_ssl_mode_are_configurable() -> None:
    settings = parts(db_port=6543, db_sslmode="verify-full")
    assert settings.database_url.endswith("@db.internal:6543/policy_rag?sslmode=verify-full")


def test_database_parts_replace_an_explicit_database_url() -> None:
    settings = parts(database_url="postgresql://other/else")
    assert "db.internal" in settings.database_url


def test_database_parts_must_be_complete() -> None:
    with pytest.raises(ValidationError, match="DB_NAME, DB_USER and DB_PASSWORD"):
        Settings(_env_file=None, db_host="db.internal")  # type: ignore[call-arg]


def test_without_a_host_the_database_url_is_left_alone() -> None:
    settings = Settings(_env_file=None, database_url="postgresql://a:b@c/d")  # type: ignore[call-arg]
    assert settings.database_url == "postgresql://a:b@c/d"


def test_the_ssl_mode_is_restricted_to_valid_values() -> None:
    with pytest.raises(ValidationError):
        parts(db_sslmode="sometimes")
