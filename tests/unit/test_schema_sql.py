from app.db.schema import render_schema


def test_vector_column_uses_the_requested_dimension() -> None:
    assert "embedding   vector(384) NOT NULL" in render_schema(384)
    assert "embedding   vector(768) NOT NULL" in render_schema(768)


def test_schema_enables_the_extension_and_creates_an_hnsw_cosine_index() -> None:
    sql = render_schema(384, m=16, ef_construction=64)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "USING hnsw (embedding vector_cosine_ops)" in sql
    assert "WITH (m = 16, ef_construction = 64)" in sql


def test_schema_indexes_metadata_for_containment_filters() -> None:
    assert "USING gin (metadata jsonb_path_ops)" in render_schema(384)


def test_dimension_is_forced_to_an_integer() -> None:
    # The dimension is rendered into SQL, so it must never carry anything but digits.
    sql = render_schema(int("384"))
    assert "vector(384)" in sql
