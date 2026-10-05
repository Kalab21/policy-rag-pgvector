"""Run the MCP server over stdio:  python -m app.mcp_server

stdout carries the protocol, so all logging goes to stderr. Point an MCP client at this command
(inside Docker: `docker compose exec -T api python -m app.mcp_server`).
"""

import logging
import sys

from app.core.config import get_settings
from app.mcp_server.server import build_server
from app.mcp_server.tools import PolicyTools
from app.services import open_services


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    settings = get_settings()
    services = open_services(settings)
    try:
        tools = PolicyTools(services.retrieval, services.rag, services.pool)
        build_server(tools, settings.mcp_tool_timeout_s).run("stdio")
    finally:
        services.close()


if __name__ == "__main__":
    main()
