"""HTTP routes."""

from fastapi import APIRouter, Request, Response

from app.db.schema import embedding_column_dim, pgvector_version
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    """Liveness plus a real database check: the pgvector extension and the vector column."""
    try:
        with request.app.state.pool.connection() as conn:
            version = pgvector_version(conn)
            dim = embedding_column_dim(conn)
    except Exception as exc:  # any database failure means "not ready"
        response.status_code = 503
        return HealthResponse(status="unavailable", detail=type(exc).__name__)

    if version is None or dim is None:
        response.status_code = 503
        return HealthResponse(
            status="unavailable",
            pgvector_version=version,
            embedding_dim=dim,
            detail="pgvector extension or chunks table is missing",
        )
    return HealthResponse(status="ok", pgvector_version=version, embedding_dim=dim)
