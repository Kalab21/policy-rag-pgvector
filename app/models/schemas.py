"""Request and response models for the HTTP API."""

from pydantic import BaseModel


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
