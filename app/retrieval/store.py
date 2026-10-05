"""Vector similarity search over the chunks table (the pgvector queries)."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from psycopg.types.json import Jsonb

from app.db.pool import DictPool
from app.models.domain import RetrievedChunk
from app.retrieval.fusion import DEFAULT_RRF_K, rrf_fuse

# Metadata keys a caller may filter on. Keys are checked against this list because they
# are used to build the JSON containment document.
FILTERABLE_KEYS = ("category", "version", "status", "document", "section")

# Approximate (HNSW) search returns only the rows its index scan visits; a selective filter
# applied afterwards can leave fewer than `top_k` rows. From pgvector 0.8 the index scan can
# continue until enough rows pass the filter ("iterative scan").
ITERATIVE_SCAN_MIN_VERSION = (0, 8, 0)


class InvalidFilterError(ValueError):
    pass


def parse_version(version: str | None) -> tuple[int, ...]:
    if not version:
        return (0,)
    return tuple(int(part) for part in version.split(".") if part.isdigit())


def supports_iterative_scan(pgvector_version: str | None) -> bool:
    return parse_version(pgvector_version) >= ITERATIVE_SCAN_MIN_VERSION


def validate_filters(filters: Mapping[str, str] | None) -> dict[str, str]:
    clean = dict(filters or {})
    unknown = sorted(set(clean) - set(FILTERABLE_KEYS))
    if unknown:
        raise InvalidFilterError(f"cannot filter on {unknown}; allowed: {list(FILTERABLE_KEYS)}")
    return clean


def _to_chunk(row: Mapping[str, Any]) -> RetrievedChunk:
    meta: dict[str, str] = row["metadata"]
    distance = float(row["distance"])
    return RetrievedChunk(
        chunk_id=int(row["id"]),
        document_id=int(row["document_id"]),
        document=meta.get("document", ""),
        title=row["source"],
        version=meta.get("version", ""),
        category=meta.get("category", ""),
        status=meta.get("status", ""),
        section=row["section"],
        text=row["chunk_text"],
        distance=distance,
        similarity=1.0 - distance,
        metadata=meta,
    )


def search_chunks(
    pool: DictPool,
    query_vector: list[float],
    top_k: int,
    filters: Mapping[str, str] | None = None,
    ef_search: int = 40,
    iterative_scan: bool = False,
) -> list[RetrievedChunk]:
    """The `top_k` chunks closest to `query_vector` by cosine distance.

    Metadata filters are applied in the same SQL statement (`metadata @> filter`), so only
    matching chunks are ranked and returned. `ORDER BY embedding <=> query` is the shape
    that lets PostgreSQL use the HNSW index.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    clean = validate_filters(filters)
    where = "WHERE metadata @> %(filter)s" if clean else ""
    # `where` is one of two fixed strings; every value is a bound parameter.
    sql = (
        "SELECT id, document_id, source, section, chunk_text, metadata,"  # nosec B608
        " embedding <=> %(query)s AS distance"
        f" FROM chunks {where}"
        " ORDER BY embedding <=> %(query)s LIMIT %(k)s"
    )
    params: dict[str, Any] = {
        "query": np.asarray(query_vector, dtype=np.float32),
        "k": top_k,
        "filter": Jsonb(clean),
    }

    with pool.connection() as conn, conn.transaction():
        # Transaction-local settings, so they never leak to other users of the connection.
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(ef_search, top_k)),))
        if iterative_scan:
            conn.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
        rows = conn.execute(sql, params).fetchall()

    return [_to_chunk(row) for row in rows]


def search_lexical(
    pool: DictPool,
    query_text: str,
    query_vector: list[float],
    top_k: int,
    filters: Mapping[str, str] | None = None,
) -> list[RetrievedChunk]:
    """The `top_k` chunks whose text matches the query's words, by PostgreSQL full-text rank.

    The question is reduced to its content lexemes (stop words dropped, English stemming) and
    a chunk matches if it contains ANY of them, so a long natural-language question still
    finds the chunk that holds its one exact term. Results are ranked by `ts_rank`, ties by
    chunk id. `query_vector` is only used to report each result's cosine distance, which the
    evidence gate needs; it does not affect which chunks match or their order.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    clean = validate_filters(filters)
    extra = "AND c.metadata @> %(filter)s" if clean else ""
    # `extra` is one of two fixed strings; every value is a bound parameter.
    sql = (
        "WITH q AS (SELECT replace(plainto_tsquery('english', %(text)s)::text, '&', '|')"  # nosec B608
        "::tsquery AS tsq)"
        " SELECT c.id, c.document_id, c.source, c.section, c.chunk_text, c.metadata,"
        " c.embedding <=> %(query)s AS distance, ts_rank(c.tsv, q.tsq) AS rank"
        f" FROM chunks c, q WHERE c.tsv @@ q.tsq {extra}"
        " ORDER BY rank DESC, c.id ASC LIMIT %(k)s"
    )
    params: dict[str, Any] = {
        "text": query_text,
        "query": np.asarray(query_vector, dtype=np.float32),
        "k": top_k,
        "filter": Jsonb(clean),
    }
    with pool.connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [replace(_to_chunk(r), score=float(r["rank"]), matched_by=("lexical",)) for r in rows]


def search_hybrid(
    pool: DictPool,
    query_text: str,
    query_vector: list[float],
    top_k: int,
    filters: Mapping[str, str] | None = None,
    candidates: int = 30,
    rrf_k: int = DEFAULT_RRF_K,
    ef_search: int = 40,
    iterative_scan: bool = False,
) -> list[RetrievedChunk]:
    """Vector search and full-text search, fused with Reciprocal Rank Fusion.

    Each retriever contributes its top `candidates` (never fewer than `top_k`), both under
    the same metadata filter; the fused list is cut to `top_k`.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    depth = max(candidates, top_k)
    semantic = search_chunks(pool, query_vector, depth, filters, ef_search, iterative_scan)
    lexical = search_lexical(pool, query_text, query_vector, depth, filters)
    return rrf_fuse({"semantic": semantic, "lexical": lexical}, rrf_k)[:top_k]
