"""Application settings, read from environment variables (and an optional local .env file)."""

from functools import lru_cache
from typing import Literal
from urllib.parse import quote

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# pgvector can index the `vector` type with HNSW up to this many dimensions.
MAX_INDEXABLE_DIM = 2000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Local development default; docker-compose and CI override it.
    database_url: str = "postgresql://policy_rag:policy_rag_local_only@localhost:5433/policy_rag"

    # Alternatively the database can be given in parts (used when a managed secret supplies the
    # user and password separately, as on AWS). If DB_HOST is set, these build DATABASE_URL and
    # replace any DATABASE_URL that was also set.
    db_host: str | None = None
    db_port: int = Field(default=5432, ge=1, le=65535)
    db_name: str | None = None
    db_user: str | None = None
    db_password: SecretStr | None = None
    db_sslmode: Literal["disable", "prefer", "require", "verify-ca", "verify-full"] = "require"

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

    # Retrieval: "semantic" (pgvector only), "lexical" (PostgreSQL full-text only) or "hybrid"
    # (both, fused with Reciprocal Rank Fusion). Each retriever contributes its top
    # `hybrid_candidates` to the fusion.
    retrieval_mode: Literal["semantic", "lexical", "hybrid"] = "semantic"
    rrf_k: int = Field(default=60, ge=1, le=1000)
    hybrid_candidates: int = Field(default=30, ge=1, le=200)

    # Cross-encoder reranking of the retrieved candidates. Off by default: it must earn that
    # place on held-out evaluation. `rerank_candidates` is how many candidates the retriever
    # hands to the cross-encoder (never the whole corpus); the final top_k is cut afterwards.
    rerank_enabled: bool = False
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(default=20, ge=1, le=100)

    # Evidence gate: the best chunk must be at least this cosine-similar to the question,
    # otherwise /api/ask refuses. Calibrated on the sample corpus (see README).
    evidence_min_similarity: float = Field(default=0.55, ge=0.0, le=1.0)
    rag_max_context_chunks: int = Field(default=4, ge=1, le=20)

    # Answer generation. "extractive" needs no model or network. "openai_compatible" calls a
    # chat-completions endpoint (OpenAI, Ollama, vLLM, ...) configured below.
    llm_provider: str = "extractive"
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_timeout_s: float = Field(default=30.0, gt=0)

    # AWS Bedrock (llm_provider="bedrock"). Credentials come from the standard AWS chain, not
    # from these settings. The model id has no default: availability is account-specific.
    bedrock_region: str | None = None
    bedrock_model_id: str | None = None
    bedrock_max_tokens: int = Field(default=512, ge=16, le=4096)
    bedrock_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    bedrock_timeout_s: float = Field(default=30.0, gt=0, le=300)
    bedrock_max_retries: int = Field(default=2, ge=0, le=5)

    # Authentication. "off" (the default) is a local/demo mode: no login, every caller sees every
    # document. "jwt" validates bearer tokens from an OIDC-style provider and limits every
    # search, answer and MCP tool to what the token's roles, tenant and departments allow.
    # Exactly one key source is needed: a JWKS URL, a PEM public key (asymmetric), or a shared
    # secret of at least 32 characters (HS256, for local demos).
    auth_mode: Literal["off", "jwt"] = "off"
    auth_issuer: str | None = None
    auth_audience: str | None = None
    auth_jwks_url: str | None = None
    auth_public_key: str | None = None
    auth_jwt_secret: SecretStr | None = None
    auth_roles_claim: str = "roles"
    auth_tenant_claim: str = "tenant_id"
    auth_department_claim: str = "department"
    auth_leeway_s: int = Field(default=30, ge=0, le=300)

    # Observability. Spans and metrics are created in-process; nothing leaves the process
    # unless OTEL_EXPORTER_OTLP_ENDPOINT is set. Prometheus text is served at /metrics.
    telemetry_enabled: bool = True
    metrics_enabled: bool = True
    otel_service_name: str = "policy-rag-platform"
    otel_exporter_otlp_endpoint: str | None = None
    # Off by default: the user's question may be sensitive. When on, a 120-character preview is
    # added to traces; the full question is never recorded.
    record_query_text: bool = False
    log_format: Literal["json", "text"] = "json"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Deadline for one MCP tool call (seconds).
    mcp_tool_timeout_s: float = Field(default=30.0, gt=0, le=300)

    db_pool_min_size: int = Field(default=1, ge=1)
    db_pool_max_size: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _database_url_from_parts(self) -> "Settings":
        if self.db_host:
            if not (self.db_name and self.db_user and self.db_password):
                raise ValueError("DB_HOST also needs DB_NAME, DB_USER and DB_PASSWORD")
            user = quote(self.db_user, safe="")
            password = quote(self.db_password.get_secret_value(), safe="")
            self.database_url = (
                f"postgresql://{user}:{password}@{self.db_host}:{self.db_port}/"
                f"{quote(self.db_name, safe='')}?sslmode={self.db_sslmode}"
            )
        return self

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
