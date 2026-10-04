"""Helpers for putting hand-made vectors into the database."""

from dataclasses import dataclass

import psycopg
from psycopg.types.json import Jsonb

from tests.conftest import TEST_DIM


def basis(index: int, *, tilt: float = 0.0, tilt_index: int = 1) -> list[float]:
    """A vector along axis `index`, optionally tilted a little towards `tilt_index`."""
    values = [0.0] * TEST_DIM
    values[index] = 1.0
    values[tilt_index] += tilt
    return values


def literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


@dataclass(frozen=True)
class Spec:
    text: str
    vector: list[float]
    category: str = "test"
    status: str = "current"
    version: str = "1.0"
    document: str = "doc"
    section: str = "Section"


def seed_chunks(url: str, specs: list[Spec]) -> None:
    """Insert one document per (document, version) and a chunk per spec, vectors as given."""
    with psycopg.connect(url) as conn:
        doc_ids: dict[tuple[str, str], int] = {}
        for i, spec in enumerate(specs):
            key = (spec.document, spec.version)
            if key not in doc_ids:
                row = conn.execute(
                    "INSERT INTO documents (name, title, version, category, status, source_path,"
                    " content_hash, embedding_model, embedding_dim)"
                    " VALUES (%s, %s, %s, %s, %s, 'x.md', 'h', 'test', %s) RETURNING id",
                    (
                        spec.document,
                        spec.document,
                        spec.version,
                        spec.category,
                        spec.status,
                        TEST_DIM,
                    ),
                ).fetchone()
                assert row is not None
                doc_ids[key] = row[0]
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, source, section, chunk_text,"
                " chunk_hash, metadata, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)",
                (
                    doc_ids[key],
                    i,
                    spec.document,
                    spec.section,
                    spec.text,
                    spec.text,
                    Jsonb(
                        {
                            "document": spec.document,
                            "title": spec.document,
                            "version": spec.version,
                            "category": spec.category,
                            "status": spec.status,
                            "section": spec.section,
                        }
                    ),
                    literal(spec.vector),
                ),
            )
