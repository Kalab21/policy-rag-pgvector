"""The policy tools, independent of the MCP framework.

Each tool is a thin, bounded, read-only wrapper over the same services the HTTP API uses
(`RetrievalService`, `RagService`, the catalog). There is no SQL, filesystem, shell or network
access here: a caller can only search the policy corpus, read a stored document, or ask a
question, and every argument is typed and length-limited before it reaches those services.
"""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from app.api.responses import ask_response, search_response
from app.db.pool import DictPool
from app.ingestion.catalog import get_document
from app.models.schemas import AskResponse, SearchFilters, SearchResponse
from app.rag.service import RagService
from app.retrieval.service import RetrievalService
from app.security.access import AccessScope

# MCP callers get a tighter bound than the HTTP API (which allows 20).
MAX_TOP_K = 10

QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
TopK = Annotated[int, Field(ge=1, le=MAX_TOP_K, description="How many chunks to return.")]
DocumentName = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{0,79}$"),
    Field(description="Document name, e.g. 'fee-schedule'."),
]
VersionText = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9]{1,3}(\.[0-9]{1,3}){0,2}$"),
    Field(description="Document version, e.g. '2.0'. Omit for the current version."),
]
RetrievalModeName = Literal["semantic", "lexical", "hybrid"]


class ToolInputError(ValueError):
    """A request the tool refuses. The message is written for the caller and is safe to show."""


class Passage(BaseModel):
    section: str
    text: str


class DocumentContent(BaseModel):
    name: str
    title: str
    version: str
    category: str
    status: str
    passages: list[Passage]
    truncated: bool = Field(description="True if the document was cut at the size limit.")


def merge_filters(caller: Mapping[str, str], enforced: Mapping[str, str]) -> dict[str, str]:
    """Combine caller filters with server-enforced ones. A caller can add filters but can never
    change or remove an enforced one: asking for a different value is an error."""
    merged = dict(caller)
    for key, value in enforced.items():
        if key in merged and merged[key] != value:
            raise ToolInputError(f"the filter {key!r} is fixed by the server and cannot be changed")
        merged[key] = value
    return merged


class PolicyTools:
    def __init__(
        self,
        retrieval: RetrievalService,
        rag: RagService,
        pool: DictPool,
        enforced_filters: Mapping[str, str] | None = None,
        access: AccessScope | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._rag = rag
        self._pool = pool
        self._enforced = dict(enforced_filters or {})
        self._access = access  # fixed for the life of the server, from a validated token

    def _filters(self, filters: SearchFilters | None) -> dict[str, str]:
        return merge_filters(filters.as_dict() if filters else {}, self._enforced)

    def search_policy(
        self,
        query: str,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        mode: RetrievalModeName | None = None,
    ) -> SearchResponse:
        applied = self._filters(filters)
        hits = self._retrieval.search(query, top_k, applied, mode, access=self._access)
        return search_response(
            query=query,
            top_k=top_k,
            filters=applied,
            mode=mode or self._retrieval.mode,
            rerank=self._retrieval.rerank_enabled,
            embedding_model=self._retrieval.model_name,
            hits=hits,
        )

    def get_policy_document(self, document: str, version: str | None = None) -> DocumentContent:
        found = get_document(self._pool, document, version, self._enforced, self._access)
        if found is None:
            raise ToolInputError("no such document (or it is not available to you)")
        return DocumentContent(**found)

    def ask_policy(self, question: str, filters: SearchFilters | None = None) -> AskResponse:
        result = self._rag.ask(question, 5, self._filters(filters), self._access)
        return ask_response(result)
