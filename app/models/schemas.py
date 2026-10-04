"""Request and response models for the HTTP API."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    pgvector_version: str | None = None
    embedding_dim: int | None = None
    detail: str | None = None
