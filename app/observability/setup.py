"""Build the application's Telemetry from settings, and install the HTTP middleware."""

import re
import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

from app.core.config import Settings
from app.observability.logs import log_event, request_id_var
from app.observability.telemetry import Telemetry, build_telemetry, span

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def telemetry_from_settings(settings: Settings) -> Telemetry:
    if not settings.telemetry_enabled:
        return Telemetry()
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics.export import MetricReader

    registry: CollectorRegistry | None = None
    readers: list[MetricReader] = []
    if settings.metrics_enabled:
        registry = CollectorRegistry()
        readers.append(PrometheusMetricReader(registry=registry))
    exporters = []
    endpoint = (settings.otel_exporter_otlp_endpoint or "").rstrip("/")
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        exporters.append(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
        readers.append(
            PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics"))
        )
    return build_telemetry(
        service_name=settings.otel_service_name,
        span_exporters=exporters,
        metric_readers=readers,
        registry=registry,
        record_query_text=settings.record_query_text,
    )


def install_http_observability(app: FastAPI, telemetry: Telemetry) -> None:
    """Request ids, one span and structured log line per request, request metrics, /metrics."""

    @app.middleware("http")
    async def observe(request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.url.path == "/metrics":
            return await call_next(request)
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        route = "unmatched"
        status = 500
        trace_id = ""
        try:
            with span(
                "http.request", {"http.method": request.method, "request.id": request_id}
            ) as sp:
                try:
                    response: Response = await call_next(request)
                except Exception as exc:
                    telemetry.record(
                        "errors", 1, {"route": route, "error_type": type(exc).__name__}
                    )
                    log_event(
                        "request.failed",
                        level=40,
                        method=request.method,
                        error_type=type(exc).__name__,
                    )
                    raise
                status = response.status_code
                matched = request.scope.get("route")
                route = getattr(matched, "path", "unmatched")
                trace_id = format(sp.get_span_context().trace_id, "032x")
                sp.set_attribute("http.route", route)
                sp.set_attribute("http.status_code", status)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            attrs = {"route": route, "method": request.method, "status_class": f"{status // 100}xx"}
            telemetry.record("requests", 1, attrs)
            telemetry.record("request_ms", elapsed_ms, {"route": route})
            if status >= 500:
                telemetry.record("errors", 1, {"route": route, "error_type": f"http_{status}"})
            # The path and status only: no query string, headers, body or user text.
            log_event(
                "request.completed",
                method=request.method,
                route=route,
                status=status,
                duration_ms=round(elapsed_ms, 1),
                trace_id=trace_id or None,
            )
            request_id_var.reset(token)

    if telemetry.registry is not None:
        registry = telemetry.registry

        @app.get("/metrics", include_in_schema=False)
        def metrics() -> Response:
            return PlainTextResponse(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
