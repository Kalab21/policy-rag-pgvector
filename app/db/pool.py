"""Connection pool with the pgvector types registered on every connection."""

from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

# A pool whose connections return rows as dicts.
DictPool = ConnectionPool[psycopg.Connection[DictRow]]


def _configure(conn: psycopg.Connection[Any]) -> None:
    register_vector(conn)
    conn.commit()  # a pool requires the connection to be idle after configuration


def create_pool(database_url: str, min_size: int = 1, max_size: int = 5) -> DictPool:
    """Open a pool. The schema (and so the vector extension) must already exist."""
    pool: DictPool = ConnectionPool(
        database_url,
        min_size=min_size,
        max_size=max_size,
        configure=_configure,
        kwargs={"row_factory": dict_row},
        open=False,
    )
    pool.open(wait=True, timeout=30)
    return pool
