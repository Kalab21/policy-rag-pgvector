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


class SearchFilters(BaseModel):
    """Metadata a search can be restricted to. Every given field must match (AND)."""

    model_config = ConfigDict(extra="forbid")

    category: str | None = Field(default=None, examples=["underwriting"])
    version: str | None = Field(default=None, examples=["2.0"])
    status: Literal["current", "superseded"] | None = Field(default=None, examples=["current"])
    document: str | None = Field(default=None, examples=["fee-schedule"])
    section: str | None = Field(default=None, examples=["Late payment fee"])

    def as_dict(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000, examples=["What is the late payment fee?"])
    top_k: int = Field(default=5, ge=1, le=20)
    filters: SearchFilters | None = None

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
    metadata: dict[str, str]


class SearchResponse(BaseModel):
    query: str
    top_k: int
    filters: dict[str, str]
    embedding_model: str
    metric: str = "cosine"
    results: list[SearchHit]
