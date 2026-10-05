"""Telemetry helpers: safe attributes, span status, redaction and the JSON log format."""

import json
import logging
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.observability.logs import JsonFormatter, TextFormatter, redact, request_id_var
from app.observability.telemetry import (
    Telemetry,
    build_telemetry,
    clean_attributes,
    set_current,
    span,
)


@pytest.fixture
def otel() -> Iterator[tuple[Telemetry, InMemorySpanExporter, InMemoryMetricReader]]:
    exporter = InMemorySpanExporter()
    reader = InMemoryMetricReader()
    telemetry = build_telemetry(
        span_exporters=[exporter], metric_readers=[reader], simple_spans=True
    )
    set_current(telemetry)
    yield telemetry, exporter, reader
    set_current(Telemetry())


def metric_total(reader: InMemoryMetricReader, name: str, **match: str) -> float:
    data = reader.get_metrics_data()
    total = 0.0
    for resource in data.resource_metrics if data else []:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                if metric.name != name:
                    continue
                for point in metric.data.data_points:
                    attrs = dict(point.attributes or {})
                    if all(attrs.get(k) == v for k, v in match.items()):
                        total += getattr(point, "value", None) or getattr(point, "sum", 0.0)
    return total


# --- attribute safety --------------------------------------------------------------------------


def test_clean_attributes_keeps_primitives_and_shortens_strings() -> None:
    cleaned = clean_attributes({"a": 1, "b": 2.5, "c": True, "d": "x" * 500, "e": ["list"]})
    assert cleaned["a"] == 1
    assert cleaned["b"] == 2.5
    assert cleaned["c"] is True
    assert len(str(cleaned["d"])) == 120
    assert isinstance(cleaned["e"], str)


@pytest.mark.parametrize(
    "key",
    [
        "authorization",
        "http.request.header.Authorization",
        "api_key",
        "AWS_SECRET_ACCESS_KEY",
        "password",
        "user.token",
        "session_cookie",
        "request.body",
        "chunk.text",
        "document.text",
        "answer.text",
    ],
)
def test_attributes_whose_names_suggest_secrets_or_content_are_dropped(key: str) -> None:
    assert clean_attributes({key: "value", "safe": "kept"}) == {"safe": "kept"}


def test_none_values_are_dropped() -> None:
    assert clean_attributes({"a": None, "b": 0}) == {"b": 0}


def test_the_question_is_recorded_by_length_only_by_default() -> None:
    question = "What is my SSN 123-45-6789?"
    attrs = Telemetry().query_attributes(question)
    assert attrs == {"query.length": len(question)}


def test_a_short_preview_is_added_only_when_enabled() -> None:
    attrs = Telemetry(record_query_text=True).query_attributes("q" * 500)
    assert attrs["query.length"] == 500
    assert len(attrs["query.preview"]) == 120


# --- spans -------------------------------------------------------------------------------------


def test_a_span_records_attributes_and_nests(otel) -> None:  # type: ignore[no-untyped-def]
    _, exporter, _ = otel
    with span("outer", {"mode": "hybrid", "password": "hunter2"}), span("inner"):
        pass
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["inner"].parent is not None
    assert spans["inner"].parent.span_id == spans["outer"].context.span_id
    assert dict(spans["outer"].attributes or {}) == {"mode": "hybrid"}


def test_a_failing_span_records_the_exception_type_not_its_message(otel) -> None:  # type: ignore[no-untyped-def]
    _, exporter, _ = otel
    with pytest.raises(RuntimeError), span("work"):
        raise RuntimeError("user said password=hunter2")
    finished = exporter.get_finished_spans()[0]
    assert finished.status.description == "RuntimeError"
    assert dict(finished.attributes or {})["error.type"] == "RuntimeError"
    assert "hunter2" not in repr(finished.to_json())
    assert not finished.events  # no recorded exception event with a message


def test_telemetry_that_is_off_does_nothing_and_does_not_fail() -> None:
    set_current(Telemetry())
    with span("anything", {"a": 1}) as sp:
        sp.set_attribute("b", 2)
    Telemetry().record("asks", 1, {"status": "answered"})


# --- metrics -----------------------------------------------------------------------------------


def test_counters_and_histograms_record_with_clean_attributes(otel) -> None:  # type: ignore[no-untyped-def]
    telemetry, _, reader = otel
    telemetry.record("asks", 1, {"status": "refused", "authorization": "Bearer abc"})
    telemetry.record("asks", 2, {"status": "answered"})
    telemetry.record("retrieval_ms", 12.5, {"mode": "hybrid"})
    assert metric_total(reader, "policy_rag.ask.total", status="refused") == 1
    assert metric_total(reader, "policy_rag.ask.total") == 3
    assert metric_total(reader, "policy_rag.retrieval.duration", mode="hybrid") == 12.5
    attrs = [
        dict(p.attributes or {})
        for r in reader.get_metrics_data().resource_metrics  # type: ignore[union-attr]
        for sc in r.scope_metrics
        for m in sc.metrics
        for p in m.data.data_points
    ]
    assert all("authorization" not in a for a in attrs)


# --- redaction ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "gone"),
    [
        ("Authorization: Bearer abcdefghijklmnop12345", "abcdefghijklmnop12345"),
        ("key AKIAIOSFODNN7EXAMPLE used", "AKIAIOSFODNN7EXAMPLE"),
        ("sk-abcdefghijklmnopqrstuv", "abcdefghijklmnopqrstuv"),
        ("password=hunter2 next", "hunter2"),
        ("api_key: s3cr3tvalue", "s3cr3tvalue"),
        ("jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.sig", "eyJhbGciOiJIUzI1NiJ9"),
    ],
)
def test_credentials_are_masked_in_log_text(raw: str, gone: str) -> None:
    assert gone not in redact(raw)
    assert "REDACTED" in redact(raw)


def test_ordinary_text_is_not_altered_by_redaction() -> None:
    assert redact("retrieved 5 chunks in 12ms") == "retrieved 5 chunks in 12ms"


# --- the JSON log line -------------------------------------------------------------------------


def make_record(message: str = "event", **fields: object) -> logging.LogRecord:
    record = logging.LogRecord("policy_rag", logging.INFO, __file__, 1, message, None, None)
    record.fields = fields  # type: ignore[attr-defined]
    return record


def test_log_lines_are_json_with_the_request_id() -> None:
    token = request_id_var.set("req-123")
    try:
        line = json.loads(
            JsonFormatter().format(make_record("rag.ask.completed", status="refused"))
        )
    finally:
        request_id_var.reset(token)
    assert line["event"] == "rag.ask.completed"
    assert line["request_id"] == "req-123"
    assert line["status"] == "refused"
    assert line["level"] == "INFO"


def test_log_lines_carry_trace_ids_inside_a_span(otel) -> None:  # type: ignore[no-untyped-def]
    with span("work"):
        line = json.loads(JsonFormatter().format(make_record()))
    assert len(line["trace_id"]) == 32
    assert len(line["span_id"]) == 16


def test_log_fields_are_filtered_and_redacted() -> None:
    record = make_record("x", token="abc", note="password=hunter2", ok=3, nothing=None)
    line = json.loads(JsonFormatter().format(record))
    assert "token" not in line
    assert "nothing" not in line
    assert line["ok"] == 3
    assert "hunter2" not in line["note"]


def test_log_lines_record_exception_types_not_messages() -> None:
    try:
        raise ValueError("secret=topsecretvalue")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "policy_rag", logging.ERROR, __file__, 1, "failed", None, sys.exc_info()
        )
    line = JsonFormatter().format(record)
    assert json.loads(line)["error_type"] == "ValueError"
    assert "topsecretvalue" not in line


def test_the_text_format_is_also_redacted() -> None:
    text = TextFormatter().format(
        make_record("login", api_key="abc", note="Bearer abcdefghijklmnop")
    )
    assert "abcdefghijklmnop" not in text
    assert "api_key" not in text
