"""Read-only listing of what has been ingested."""

from collections.abc import Mapping
from typing import Any

from app.db.pool import DictPool


def list_documents(pool: DictPool) -> list[dict[str, Any]]:
    with pool.connection() as conn:
        return list(
            conn.execute(
                "SELECT d.name, d.title, d.version, d.category, d.status, d.embedding_model,"
                " d.embedding_dim, count(c.id)::int AS chunks"
                " FROM documents d LEFT JOIN chunks c ON c.document_id = d.id"
                " GROUP BY d.id ORDER BY d.name, d.version"
            )
        )


MAX_DOCUMENT_CHARS = 60_000  # a read tool must never return an unbounded amount of text


def get_document(
    pool: DictPool,
    name: str,
    version: str | None = None,
    filters: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """One stored document with its passages in reading order, or None if it does not exist.

    Without a version, the current version is preferred. `filters` can only narrow the choice
    (category, status): it is how a caller's access restrictions apply to this read.
    """
    clauses = ["d.name = %(name)s"]
    params: dict[str, Any] = {"name": name}
    if version is not None:
        clauses.append("d.version = %(version)s")
        params["version"] = version
    for key in ("category", "status"):
        if filters and key in filters:
            clauses.append(f"d.{key} = %({key})s")  # nosec B608 - key is from a fixed tuple
            params[key] = filters[key]
    sql = (
        "SELECT d.id, d.name, d.title, d.version, d.category, d.status FROM documents d"  # nosec B608
        f" WHERE {' AND '.join(clauses)}"
        " ORDER BY (d.status = 'current') DESC, d.version DESC LIMIT 1"
    )
    with pool.connection() as conn:
        doc = conn.execute(sql, params).fetchone()
        if doc is None:
            return None
        rows = conn.execute(
            "SELECT section, chunk_text FROM chunks WHERE document_id = %s"
            " ORDER BY chunk_index LIMIT 500",
            (doc["id"],),
        ).fetchall()
    passages: list[dict[str, str]] = []
    used = 0
    truncated = False
    for row in rows:
        if used + len(row["chunk_text"]) > MAX_DOCUMENT_CHARS:
            truncated = True
            break
        used += len(row["chunk_text"])
        passages.append({"section": row["section"], "text": row["chunk_text"]})
    return {
        "name": doc["name"],
        "title": doc["title"],
        "version": doc["version"],
        "category": doc["category"],
        "status": doc["status"],
        "passages": passages,
        "truncated": truncated,
    }
