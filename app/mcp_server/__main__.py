"""Run the MCP server over stdio:  python -m app.mcp_server

stdout carries the protocol, so all logging goes to stderr. Point an MCP client at this command
(inside Docker: `docker compose exec -T api python -m app.mcp_server`). Traces and metrics are
exported only if OTEL_EXPORTER_OTLP_ENDPOINT is set; there is no /metrics endpoint here.

Authorization: with AUTH_MODE=jwt the process refuses to start without a valid MCP_ACCESS_TOKEN
in its environment (the client that launches it supplies its own credential), and every tool
call is limited to what that token's roles, tenant and departments allow, exactly as for the
HTTP API. With AUTH_MODE=off (local demo) there is no restriction.
"""

import os
import sys

from app.core.config import get_settings
from app.mcp_server.server import build_server
from app.mcp_server.tools import PolicyTools
from app.observability.logs import configure_logging
from app.observability.setup import telemetry_from_settings
from app.observability.telemetry import set_current
from app.security.access import AccessScope
from app.security.auth import AuthError, TokenValidator
from app.services import open_services


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_format, settings.log_level)
    scope: AccessScope | None = None
    if settings.auth_mode == "jwt":
        token = os.environ.get("MCP_ACCESS_TOKEN", "")
        try:
            scope = TokenValidator(settings).scope(token)
        except AuthError as exc:
            print(f"MCP server not started: {exc.detail} ({exc.code})", file=sys.stderr)
            sys.exit(2)
    telemetry = telemetry_from_settings(settings)
    set_current(telemetry)
    services = open_services(settings)
    try:
        tools = PolicyTools(services.retrieval, services.rag, services.pool, access=scope)
        build_server(tools, settings.mcp_tool_timeout_s).run("stdio")
    finally:
        services.close()
        telemetry.shutdown()


if __name__ == "__main__":
    main()
