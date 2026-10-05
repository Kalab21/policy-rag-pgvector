"""Database schema: documents, chunks, the pgvector column and its indexes."""

from typing import Any

import psycopg

# `vector(N)` cannot be a bound parameter, so the dimension is rendered into the SQL after
# being forced to an int. Everything else is static.
_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name            TEXT        NOT NULL,
    title           TEXT        NOT NULL,
    version         TEXT        NOT NULL,
    category        TEXT        NOT NULL,
    status          TEXT        NOT NULL CHECK (status IN ('current', 'superseded')),
    source_path     TEXT        NOT NULL,
    content_hash    TEXT        NOT NULL,
    embedding_model TEXT        NOT NULL,
    embedding_dim   INTEGER     NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id BIGINT      NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index INTEGER     NOT NULL,
    source      TEXT        NOT NULL,
    section     TEXT        NOT NULL,
    chunk_text  TEXT        NOT NULL,
    chunk_hash  TEXT        NOT NULL,
    metadata    JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
    embedding   vector({dim}) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- Full-text search vector over the title, section and text. A generated column keeps it in
-- sync with the row; ADD COLUMN IF NOT EXISTS upgrades tables created before it existed.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english', source || ' ' || section || ' ' || chunk_text)
    ) STORED;
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);

-- Metadata filters use the containment operator (metadata @> '{{"category": "fees"}}').
CREATE INDEX IF NOT EXISTS chunks_metadata_gin ON chunks USING gin (metadata jsonb_path_ops);

-- Approximate nearest-neighbour index for cosine distance (the <=> operator).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = {m}, ef_construction = {ef_construction});
"""


def render_schema(dim: int, m: int = 16, ef_construction: int = 64) -> str:
    """The schema SQL for a vector column of `dim` dimensions."""
    return _SCHEMA.format(dim=int(dim), m=int(m), ef_construction=int(ef_construction))


def init_schema(database_url: str, dim: int, m: int = 16, ef_construction: int = 64) -> None:
    """Create the extension, tables and indexes if they do not exist (idempotent).

    Uses its own connection because the extension must exist before the pgvector types
    can be registered on pooled connections.
    """
    with psycopg.connect(database_url) as conn:
        conn.execute(render_schema(dim, m, ef_construction))


def embedding_column_dim(conn: psycopg.Connection[Any]) -> int | None:
    """Width of chunks.embedding, or None if the table does not exist."""
    row = conn.execute(
        """
        SELECT a.atttypmod
        FROM pg_attribute a
        WHERE a.attrelid = to_regclass('chunks') AND a.attname = 'embedding' AND NOT a.attisdropped
        """
    ).fetchone()
    if row is None:
        return None
    value = row["atttypmod"] if isinstance(row, dict) else row[0]
    return int(value)


def pgvector_version(conn: psycopg.Connection[Any]) -> str | None:
    """Installed pgvector extension version, or None if it is not installed."""
    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    if row is None:
        return None
    return str(row["extversion"] if isinstance(row, dict) else row[0])
