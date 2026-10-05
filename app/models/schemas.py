"""Request and response models for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentInfo(BaseModel):
    name: str
    title: str
    version: str
    category: str
    status: str
    embedding_model: str
    embedding_dim: int
    chunks: int


class HealthResponse(BaseModel):
    status: str
    pgvector_version: str | None = None
    embedding_dim: int | None = None
    detail: str | None = None


class MeResponse(BaseModel):
    auth_mode: Literal["off", "jwt"]
    authenticated: bool
    roles: list[str] = []
    tenant_id: str | None = None
    departments: list[str] = []
    max_access_level: str | None = None
    note: str | None = None


class SearchFilters(BaseModel):
    """Metadata a search can be restricted to. Every given field must match (AND)."""

    model_config = ConfigDict(extra="forbid")

    category: str | None = Field(default=None, max_length=200, examples=["underwriting"])
    version: str | None = Field(default=None, max_length=50, examples=["2.0"])
    status: Literal["current", "superseded"] | None = Field(default=None, examples=["current"])
    document: str | None = Field(default=None, max_length=200, examples=["fee-schedule"])
    section: str | None = Field(default=None, max_length=200, examples=["Late payment fee"])

    def as_dict(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000, examples=["What is the late payment fee?"])
    top_k: int = Field(default=5, ge=1, le=20)
    filters: SearchFilters | None = None
    mode: Literal["semantic", "lexical", "hybrid"] | None = Field(
        default=None, description="Defaults to the server's RETRIEVAL_MODE."
    )
    rerank: bool | None = Field(
        default=None,
        description="Cross-encoder reranking. Defaults to the server's RERANK_ENABLED.",
    )

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class SearchHit(BaseModel):
    chunk_id: int
    document: str
    title: str
    version: str
    category: str
    status: str
    section: str
    text: str
    distance: float
    similarity: float
    score: float | None = None
    rerank_score: float | None = None
    matched_by: list[str] = []
    metadata: dict[str, str]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000, examples=["What is the late payment fee?"])
    top_k: int = Field(default=5, ge=1, le=20)
    filters: SearchFilters | None = None

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class EvidenceInfo(BaseModel):
    status: Literal["sufficient", "insufficient"]
    reason: str
    best_similarity: float | None
    threshold: float
    chunks_considered: int
    chunks_used: int


class Citation(BaseModel):
    citation: int
    chunk_id: int
    document: str
    title: str
    version: str
    category: str
    status: str
    section: str
    similarity: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    status: Literal["answered", "refused"]
    refusal_reason: str | None
    evidence: EvidenceInfo
    sources: list[Citation]
    retrieved_chunk_ids: list[int]
    generator: str
    embedding_model: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    filters: dict[str, str]
    mode: str
    rerank: bool
    embedding_model: str
    metric: str = "cosine"
    results: list[SearchHit]
