"""Application settings, read from environment variables (and an optional local .env file)."""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# pgvector can index the `vector` type with HNSW up to this many dimensions.
MAX_INDEXABLE_DIM = 2000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Local development default; docker-compose and CI override it.
    database_url: str = "postgresql://policy_rag:policy_rag_local_only@localhost:5433/policy_rag"

    # The embedding model and the width of the vector column must agree. The column width
    # is checked against this value at startup.
    embedding_provider: str = "fastembed"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = Field(default=384, gt=0, le=MAX_INDEXABLE_DIM)

    # Chunking (characters). The overlap must be smaller than the chunk size.
    chunk_size: int = Field(default=700, ge=100, le=2000)
    chunk_overlap: int = Field(default=120, ge=0)
    sample_data_dir: str = "sample_data/policies"

    # Create tables and indexes on startup when they are missing.
    auto_init_schema: bool = True

    # HNSW index parameters (build time) and search width (query time).
    hnsw_m: int = Field(default=16, ge=2, le=100)
    hnsw_ef_construction: int = Field(default=64, ge=4, le=1000)
    hnsw_ef_search: int = Field(default=40, ge=1, le=1000)

    db_pool_min_size: int = Field(default=1, ge=1)
    db_pool_max_size: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
