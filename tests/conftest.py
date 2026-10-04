"""Shared fixtures.

Integration tests run against a real PostgreSQL with the pgvector extension. Point
TEST_DATABASE_URL at a database that is safe to wipe; the tests drop and recreate the
application tables. Without it, integration tests are skipped.
"""

import os

import psycopg
import pytest

from app.db.schema import init_schema

TEST_DIM = 384


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set; start PostgreSQL with pgvector to run this")
    return url


@pytest.fixture
def clean_db(database_url: str) -> str:
    """An empty database with the schema freshly created."""
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, documents CASCADE")
    init_schema(database_url, TEST_DIM)
    return database_url
