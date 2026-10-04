"""The schema against a real PostgreSQL with pgvector."""

import psycopg
import pytest
from psycopg.rows import dict_row

from app.db.schema import embedding_column_dim, init_schema, pgvector_version
from tests.conftest import TEST_DIM

pytestmark = pytest.mark.integration


def _conn(url: str) -> psycopg.Connection[dict[str, object]]:
    return psycopg.connect(url, row_factory=dict_row)


def test_extension_is_installed(clean_db: str) -> None:
    with _conn(clean_db) as conn:
        version = pgvector_version(conn)
    assert version is not None
    assert version[0].isdigit()


def test_embedding_column_has_the_configured_width(clean_db: str) -> None:
    with _conn(clean_db) as conn:
        assert embedding_column_dim(conn) == TEST_DIM


def test_init_schema_is_idempotent(clean_db: str) -> None:
    init_schema(clean_db, TEST_DIM)
    init_schema(clean_db, TEST_DIM)
    with _conn(clean_db) as conn:
        tables = {
            r["tablename"]
            for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        }
    assert {"documents", "chunks"} <= tables


def test_hnsw_cosine_index_and_metadata_gin_index_exist(clean_db: str) -> None:
    with _conn(clean_db) as conn:
        defs = {
            r["indexname"]: r["indexdef"]
            for r in conn.execute(
                "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'chunks'"
            )
        }
    assert "USING hnsw (embedding vector_cosine_ops)" in defs["chunks_embedding_hnsw"]
    assert "USING gin (metadata jsonb_path_ops)" in defs["chunks_metadata_gin"]


def test_pgvector_cosine_distance_operator_works(clean_db: str) -> None:
    """A real vector round trip: store three vectors, order them by cosine distance."""
    vectors = {
        "x": [1.0] + [0.0] * (TEST_DIM - 1),
        "y": [0.0, 1.0] + [0.0] * (TEST_DIM - 2),
        "near_x": [1.0, 0.1] + [0.0] * (TEST_DIM - 2),
    }
    with _conn(clean_db) as conn:
        doc_id = conn.execute(
            "INSERT INTO documents (name, title, version, category, status, source_path,"
            " content_hash, embedding_model, embedding_dim)"
            " VALUES ('d', 'D', '1', 'test', 'current', 'd.md', 'h', 'm', %s) RETURNING id",
            (TEST_DIM,),
        ).fetchone()
        assert doc_id is not None
        for i, (label, vec) in enumerate(vectors.items()):
            literal = "[" + ",".join(str(v) for v in vec) + "]"
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, source, section, chunk_text,"
                " chunk_hash, embedding) VALUES (%s, %s, 'D', 's', %s, %s, %s::vector)",
                (doc_id["id"], i, label, label, literal),
            )
        query = "[" + ",".join(["1"] + ["0"] * (TEST_DIM - 1)) + "]"
        rows = conn.execute(
            "SELECT chunk_text, embedding <=> %s::vector AS distance FROM chunks ORDER BY distance",
            (query,),
        ).fetchall()
    assert [r["chunk_text"] for r in rows] == ["x", "near_x", "y"]
    assert rows[0]["distance"] == pytest.approx(0.0, abs=1e-9)
    assert rows[2]["distance"] == pytest.approx(1.0, abs=1e-9)
