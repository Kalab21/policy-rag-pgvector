"""Read-only listing of what has been ingested."""

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
