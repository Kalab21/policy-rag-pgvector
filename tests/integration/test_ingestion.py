"""Ingestion into a real PostgreSQL with pgvector (the embedder is a deterministic fake)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from psycopg_pool import ConnectionPool

from app.core.config import Settings
from app.db.pool import create_pool
from app.ingestion.loader import load_documents, parse_document
from app.ingestion.pipeline import ingest_documents
from app.main import create_app
from tests.conftest import TEST_DIM
from tests.fakes import FailingEmbedder, HashingEmbedder

pytestmark = pytest.mark.integration

SIZE, OVERLAP = 400, 80
SAMPLES = Path("sample_data/policies")

DOC = (
    '---\nname: demo\ntitle: Demo\nversion: "1.0"\ncategory: fees\nstatus: current\n---\n\n'
    "# Demo\n\n## Late fee\nThe late fee is $35.\n\n## Refunds\nRefunds take ten days.\n"
)


@pytest.fixture
def pool(clean_db: str):  # type: ignore[no-untyped-def]
    pool = create_pool(clean_db)
    yield pool
    pool.close()


def _one(pool: ConnectionPool, sql: str, *params: object) -> dict[str, object]:
    with pool.connection() as conn:
        row = conn.execute(sql, params).fetchone()  # type: ignore[arg-type]
    assert row is not None
    return dict(row)


def test_documents_and_chunks_are_stored_with_vectors(pool: ConnectionPool) -> None:
    docs = load_documents(SAMPLES)
    results = ingest_documents(pool, HashingEmbedder(), docs, SIZE, OVERLAP)

    assert {r.action for r in results} == {"created"}
    assert _one(pool, "SELECT count(*) AS n FROM documents")["n"] == len(docs)
    assert _one(pool, "SELECT count(*) AS n FROM chunks")["n"] == sum(r.chunks for r in results)
    # The stored vectors really have the configured width.
    assert _one(pool, "SELECT vector_dims(embedding) AS d FROM chunks LIMIT 1")["d"] == TEST_DIM


def test_the_embedding_model_and_width_are_recorded_per_document(pool: ConnectionPool) -> None:
    ingest_documents(pool, HashingEmbedder(), [parse_document(DOC, "demo.md")], SIZE, OVERLAP)
    row = _one(pool, "SELECT embedding_model, embedding_dim, content_hash FROM documents")
    assert row["embedding_model"] == "test-hashing-embedder"
    assert row["embedding_dim"] == TEST_DIM
    assert len(str(row["content_hash"])) == 64


def test_chunk_metadata_is_stored_as_jsonb(pool: ConnectionPool) -> None:
    ingest_documents(pool, HashingEmbedder(), [parse_document(DOC, "demo.md")], SIZE, OVERLAP)
    row = _one(
        pool,
        "SELECT metadata, section, source FROM chunks WHERE metadata @> %s::jsonb LIMIT 1",
        '{"category": "fees", "status": "current"}',
    )
    assert row["metadata"]["document"] == "demo"  # type: ignore[index]
    assert row["source"] == "Demo"


def test_running_ingestion_twice_is_idempotent_and_does_not_re_embed(pool: ConnectionPool) -> None:
    docs = load_documents(SAMPLES)
    embedder = HashingEmbedder()
    first = ingest_documents(pool, embedder, docs, SIZE, OVERLAP)
    ids_before = _one(pool, "SELECT array_agg(id ORDER BY id) AS ids FROM chunks")["ids"]
    calls_after_first = embedder.calls

    second = ingest_documents(pool, embedder, docs, SIZE, OVERLAP)

    assert {r.action for r in second} == {"skipped"}
    assert [r.chunks for r in second] == [r.chunks for r in first]
    assert embedder.calls == calls_after_first  # nothing was embedded the second time
    assert _one(pool, "SELECT array_agg(id ORDER BY id) AS ids FROM chunks")["ids"] == ids_before
    assert _one(pool, "SELECT count(*) AS n FROM documents")["n"] == len(docs)


def test_a_changed_document_replaces_its_chunks(pool: ConnectionPool) -> None:
    embedder = HashingEmbedder()
    ingest_documents(pool, embedder, [parse_document(DOC, "demo.md")], SIZE, OVERLAP)

    changed = parse_document(DOC.replace("$35", "$40"), "demo.md")
    result = ingest_documents(pool, embedder, [changed], SIZE, OVERLAP)

    assert result[0].action == "updated"
    texts = [
        r["chunk_text"] for r in _all(pool, "SELECT chunk_text FROM chunks ORDER BY chunk_index")
    ]
    assert "The late fee is $40." in texts
    assert "The late fee is $35." not in texts
    assert _one(pool, "SELECT count(*) AS n FROM documents")["n"] == 1


def test_changing_the_chunk_size_re_ingests_the_document(pool: ConnectionPool) -> None:
    doc = parse_document(DOC, "demo.md")
    ingest_documents(pool, HashingEmbedder(), [doc], SIZE, OVERLAP)
    result = ingest_documents(pool, HashingEmbedder(), [doc], SIZE + 100, OVERLAP)
    assert result[0].action == "updated"


def test_the_same_name_can_exist_in_two_versions(pool: ConnectionPool) -> None:
    ingest_documents(pool, HashingEmbedder(), load_documents(SAMPLES), SIZE, OVERLAP)
    rows = _all(pool, "SELECT version, status FROM documents WHERE name = 'underwriting-policy'")
    assert {(r["version"], r["status"]) for r in rows} == {
        ("1.0", "superseded"),
        ("2.0", "current"),
    }


def test_a_failed_embedding_leaves_existing_data_untouched(pool: ConnectionPool) -> None:
    ingest_documents(pool, HashingEmbedder(), [parse_document(DOC, "demo.md")], SIZE, OVERLAP)
    before = _one(pool, "SELECT count(*) AS n FROM chunks")["n"]

    changed = parse_document(DOC.replace("$35", "$99"), "demo.md")
    with pytest.raises(RuntimeError, match="unavailable"):
        ingest_documents(pool, FailingEmbedder(), [changed], SIZE, OVERLAP)

    assert _one(pool, "SELECT count(*) AS n FROM chunks")["n"] == before
    texts = [r["chunk_text"] for r in _all(pool, "SELECT chunk_text FROM chunks")]
    assert "The late fee is $35." in texts


def test_documents_endpoint_lists_what_was_ingested(clean_db: str) -> None:
    settings = Settings(database_url=clean_db, embedding_dim=TEST_DIM, _env_file=None)  # type: ignore[call-arg]
    with TestClient(create_app(settings)) as client:
        ingest_documents(
            client.app.state.pool,  # type: ignore[attr-defined]
            HashingEmbedder(),
            load_documents(SAMPLES),
            SIZE,
            OVERLAP,
        )
        body = client.get("/api/documents").json()
    assert len(body) == 7
    fee = next(d for d in body if d["name"] == "fee-schedule")
    assert fee["category"] == "fees"
    assert fee["chunks"] > 0
    assert fee["embedding_dim"] == TEST_DIM


def _all(pool: ConnectionPool, sql: str) -> list[dict[str, object]]:
    with pool.connection() as conn:
        return [dict(r) for r in conn.execute(sql)]  # type: ignore[arg-type]
