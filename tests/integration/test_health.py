import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.conftest import TEST_DIM

pytestmark = pytest.mark.integration


def _settings(url: str, dim: int = TEST_DIM) -> Settings:
    return Settings(database_url=url, embedding_dim=dim, _env_file=None)  # type: ignore[call-arg]


def test_health_reports_pgvector_and_vector_width(clean_db: str) -> None:
    with TestClient(create_app(_settings(clean_db))) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["embedding_dim"] == TEST_DIM
    assert body["pgvector_version"]


def test_startup_fails_fast_when_the_model_width_does_not_match_the_table(clean_db: str) -> None:
    app = create_app(
        _settings(clean_db, dim=TEST_DIM + 1).model_copy(update={"auto_init_schema": False})
    )
    with pytest.raises(RuntimeError, match="dimensions"), TestClient(app):
        pass
