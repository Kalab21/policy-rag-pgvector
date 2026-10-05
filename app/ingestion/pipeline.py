"""Ingestion: parse, chunk, embed and store documents, skipping what has not changed."""

import hashlib
import json
from collections.abc import Sequence
from typing import Literal

import numpy as np
from psycopg.types.json import Jsonb

from app.db.pool import DictPool
from app.embeddings.base import EmbeddingProvider
from app.ingestion.chunking import chunk_document
from app.models.domain import IngestResult, ParsedDocument


def document_fingerprint(
    doc: ParsedDocument, model_name: str, chunk_size: int, chunk_overlap: int
) -> str:
    """Hash of everything that determines a document's stored chunks and vectors.

    If the fingerprint of the stored document equals the new one, re-embedding would
    produce the same rows, so ingestion skips it.
    """
    payload = json.dumps(
        {
            "name": doc.name,
            "title": doc.title,
            "version": doc.version,
            "category": doc.category,
            "status": doc.status,
            "tenant_id": doc.tenant_id,
            "department": doc.department,
            "access_level": doc.access_level,
            "body": doc.body,
            "model": model_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ingest_documents(
    pool: DictPool,
    provider: EmbeddingProvider,
    documents: Sequence[ParsedDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[IngestResult]:
    """Store `documents`. Safe to run repeatedly.

    - unchanged document (same fingerprint): skipped, nothing is embedded
    - changed document with the same name and version: its chunks are replaced atomically
    - new document: created
    """
    results: list[IngestResult] = []
    for doc in documents:
        fingerprint = document_fingerprint(doc, provider.model_name, chunk_size, chunk_overlap)
        with pool.connection() as conn:
            existing = conn.execute(
                "SELECT id, content_hash FROM documents WHERE name = %s AND version = %s",
                (doc.name, doc.version),
            ).fetchone()

            if existing is not None and existing["content_hash"] == fingerprint:
                count = conn.execute(
                    "SELECT count(*) AS n FROM chunks WHERE document_id = %s", (existing["id"],)
                ).fetchone()
                assert count is not None
                results.append(IngestResult(doc.name, doc.version, "skipped", int(count["n"])))
                continue

            chunks = chunk_document(doc, chunk_size, chunk_overlap)
            vectors = provider.embed_documents([c.embed_text for c in chunks])
            if len(vectors) != len(chunks):
                raise RuntimeError("the embedding provider returned the wrong number of vectors")

            with conn.transaction():
                if existing is not None:
                    conn.execute("DELETE FROM documents WHERE id = %s", (existing["id"],))
                row = conn.execute(
                    "INSERT INTO documents (name, title, version, category, status, source_path,"
                    " content_hash, embedding_model, embedding_dim, tenant_id, department,"
                    " access_level)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                    (
                        doc.name,
                        doc.title,
                        doc.version,
                        doc.category,
                        doc.status,
                        doc.source_path,
                        fingerprint,
                        provider.model_name,
                        provider.dimension,
                        doc.tenant_id,
                        doc.department,
                        doc.access_level,
                    ),
                ).fetchone()
                assert row is not None
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chunks (document_id, chunk_index, source, section,"
                        " chunk_text, chunk_hash, metadata, embedding)"
                        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        [
                            (
                                row["id"],
                                c.index,
                                doc.title,
                                c.section,
                                c.text,
                                c.text_hash,
                                Jsonb(c.metadata),
                                np.asarray(vec, dtype=np.float32),
                            )
                            for c, vec in zip(chunks, vectors, strict=True)
                        ],
                    )
            action: Literal["created", "updated"] = "updated" if existing is not None else "created"
            results.append(IngestResult(doc.name, doc.version, action, len(chunks)))
    return results
