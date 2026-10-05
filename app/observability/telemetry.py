"""OpenTelemetry tracing and metrics for the RAG pipeline.

Design rules:
- Spans carry only safe attributes: counts, modes, scores, thresholds, status, provider and
  model names, latencies. Never document text, answers, credentials or tokens, and not the
  user's question (only its length) unless RECORD_QUERY_TEXT is turned on.
- Telemetry is optional. With no exporter configured it costs almost nothing, and nothing leaves
  the process. Prometheus text is served at /metrics; OTLP export happens only if an endpoint
  is configured.
- Each `Telemetry` owns its providers (no process-global state), so an app, or a test, can
  create and discard one freely.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import Span, Status, StatusCode, Tracer
from prometheus_client import CollectorRegistry

# Attribute names containing any of these are dropped, whatever their value.
FORBIDDEN_KEY_PARTS = (
    "token",
    "authorization",
    "api_key",
    "apikey",
    "password",
    "secret",
    "credential",
    "cookie",
    "body",
    "document.text",
    "chunk.text",
    "answer.text",
)
MAX_ATTRIBUTE_CHARS = 120


def clean_attributes(attributes: Mapping[str, Any] | None) -> dict[str, str | bool | int | float]:
    """Keep only safe, primitive, short attributes."""
    cleaned: dict[str, str | bool | int | float] = {}
    for key, value in (attributes or {}).items():
        lowered = key.lower()
        if any(part in lowered for part in FORBIDDEN_KEY_PARTS) or value is None:
            continue
        if isinstance(value, bool | int | float):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)[:MAX_ATTRIBUTE_CHARS]
    return cleaned


class Telemetry:
    def __init__(
        self,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
        registry: CollectorRegistry | None = None,
        record_query_text: bool = False,
    ) -> None:
        self._tracer_provider = tracer_provider
        self._meter_provider = meter_provider
        self.registry = registry
        self.record_query_text = record_query_text
        self.tracer: Tracer = (
            tracer_provider.get_tracer("policy_rag")
            if tracer_provider is not None
            else trace.NoOpTracer()
        )
        self._instruments_ready = meter_provider is not None
        if meter_provider is not None:
            meter = meter_provider.get_meter("policy_rag")
            self._create_instruments(meter)

    # --- instruments -----------------------------------------------------------------------

    def _create_instruments(self, meter: Meter) -> None:
        self.requests = meter.create_counter("policy_rag.requests", description="HTTP requests")
        self.errors = meter.create_counter("policy_rag.errors", description="Failed requests")
        self.request_ms = meter.create_histogram("policy_rag.request.duration", unit="ms")
        self.asks = meter.create_counter("policy_rag.ask.total", description="Ask requests")
        self.refusals = meter.create_counter("policy_rag.ask.refusals", description="Refusals")
        self.retrieval_ms = meter.create_histogram("policy_rag.retrieval.duration", unit="ms")
        self.rerank_ms = meter.create_histogram("policy_rag.rerank.duration", unit="ms")
        self.generation_ms = meter.create_histogram("policy_rag.generation.duration", unit="ms")
        self.returned_chunks = meter.create_histogram("policy_rag.retrieval.returned_chunks")
        self.provider_errors = meter.create_counter("policy_rag.provider.errors")
        self.mcp_calls = meter.create_counter("policy_rag.mcp.tool.calls")
        self.mcp_failures = meter.create_counter("policy_rag.mcp.tool.failures")

    def record(
        self, instrument: str, value: float, attributes: Mapping[str, Any] | None = None
    ) -> None:
        """Record on a named instrument (counter or histogram); a no-op when metrics are off."""
        if not self._instruments_ready:
            return
        target = getattr(self, instrument)
        attrs = clean_attributes(attributes)
        if hasattr(target, "add"):
            target.add(value, attrs)
        else:
            target.record(value, attrs)

    def shutdown(self) -> None:
        if self._tracer_provider is not None:
            self._tracer_provider.shutdown()
        if self._meter_provider is not None:
            self._meter_provider.shutdown()

    def query_attributes(self, text: str) -> dict[str, Any]:
        """The question is described by its length; its text only if explicitly enabled."""
        attrs: dict[str, Any] = {"query.length": len(text)}
        if self.record_query_text:
            attrs["query.preview"] = text[:MAX_ATTRIBUTE_CHARS]
        return attrs


def build_telemetry(
    *,
    service_name: str = "policy-rag-platform",
    service_version: str = "0.1.0",
    span_exporters: Sequence[SpanExporter] | None = None,
    metric_readers: Sequence[MetricReader] | None = None,
    registry: CollectorRegistry | None = None,
    record_query_text: bool = False,
    simple_spans: bool = False,
) -> Telemetry:
    """Assemble providers from explicit exporters and readers."""
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    resource = Resource.create({"service.name": service_name, "service.version": service_version})
    tracer_provider = TracerProvider(resource=resource)
    for exporter in span_exporters or []:
        processor = SimpleSpanProcessor(exporter) if simple_spans else BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(processor)
    meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers or [])
    return Telemetry(tracer_provider, meter_provider, registry, record_query_text)


_current = Telemetry()  # a no-op until the application installs a real one


def current() -> Telemetry:
    return _current


def set_current(telemetry: Telemetry) -> None:
    global _current
    _current = telemetry


@contextmanager
def span(name: str, attributes: Mapping[str, Any] | None = None) -> Iterator[Span]:
    """A span under the current telemetry. On error the status is set from the exception *type*
    only, because exception messages can contain user input or secrets."""
    telemetry = current()
    with telemetry.tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as active:
        for key, value in clean_attributes(attributes).items():
            active.set_attribute(key, value)
        try:
            yield active
        except Exception as exc:
            active.set_attribute("error.type", type(exc).__name__)
            active.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise


def set_attributes(active: Span, attributes: Mapping[str, Any]) -> None:
    for key, value in clean_attributes(attributes).items():
        active.set_attribute(key, value)
