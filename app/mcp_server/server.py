"""The MCP server: three read-only policy tools over stdio.

stdio only, on purpose. A tool server reachable over the network would need authentication,
which this project does not have yet; a client that launches the process locally inherits the
caller's own trust. Tool failures are reported as MCP tool errors with a safe message: schema
violations are rejected before any code runs, expected failures carry a short explanation, and
anything unexpected is masked by the SDK as a generic error.
"""

import logging
from collections.abc import Callable
from typing import TypeVar

import anyio
import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from app.mcp_server.tools import (
    DocumentContent,
    DocumentName,
    PolicyTools,
    QueryText,
    RetrievalModeName,
    ToolInputError,
    TopK,
    VersionText,
)
from app.models.schemas import AskResponse, SearchFilters, SearchResponse
from app.rag.generator import GenerationError
from app.retrieval.store import InvalidFilterError

logger = logging.getLogger(__name__)
T = TypeVar("T")

READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


def build_server(tools: PolicyTools, timeout_s: float = 30.0) -> MCPServer:
    server = MCPServer(
        "policy-rag",
        instructions=(
            "Read-only access to a small set of synthetic lending-policy documents. Use "
            "search_policy to find passages, get_policy_document to read one document, and "
            "ask_policy for a cited answer that refuses when the documents do not support one."
        ),
    )

    async def run(work: Callable[[], T]) -> T:
        """Run blocking service code off the event loop, with a deadline and safe errors."""
        try:
            with anyio.fail_after(timeout_s):
                return await anyio.to_thread.run_sync(work, abandon_on_cancel=True)
        except TimeoutError:
            raise ToolError("the request timed out") from None
        except (ToolInputError, InvalidFilterError, ValueError) as exc:
            raise ToolError(str(exc)) from None
        except (GenerationError, httpx.HTTPError):
            raise ToolError("the answer generator is unavailable") from None

    @server.tool(
        name="search_policy",
        description=(
            "Search the policy documents. Returns the best-matching passages with their "
            "document, version, section and similarity. Optional metadata filters narrow the "
            "search; 'mode' picks semantic, lexical or hybrid retrieval."
        ),
        annotations=READ_ONLY,
    )
    async def search_policy(
        query: QueryText,
        top_k: TopK = 5,
        filters: SearchFilters | None = None,
        mode: RetrievalModeName | None = None,
    ) -> SearchResponse:
        return await run(lambda: tools.search_policy(query, top_k, filters, mode))

    @server.tool(
        name="get_policy_document",
        description=(
            "Read one stored policy document in full, by name and optionally version. Without "
            "a version, the current one is returned."
        ),
        annotations=READ_ONLY,
    )
    async def get_policy_document(
        document: DocumentName, version: VersionText | None = None
    ) -> DocumentContent:
        return await run(lambda: tools.get_policy_document(document, version))

    @server.tool(
        name="ask_policy",
        description=(
            "Ask a question about the policies. Returns a cited answer, or a refusal when the "
            "retrieved evidence is not strong enough. Answers use current policy unless the "
            "status filter says otherwise."
        ),
        annotations=READ_ONLY,
    )
    async def ask_policy(question: QueryText, filters: SearchFilters | None = None) -> AskResponse:
        return await run(lambda: tools.ask_policy(question, filters))

    return server
