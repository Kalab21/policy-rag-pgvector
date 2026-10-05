"""Structured (JSON) logging with request correlation and redaction.

One logger, `policy_rag`, writes one JSON object per line to stderr. Every line carries the
request id and, when a span is active, its trace and span ids, so a log line can be matched to
its trace. Field values are redacted: secrets that look like credentials are masked, and keys
that name secrets are dropped. Log events describe what happened (status, counts, scores,
latency); they do not contain the user's question, document text or answers.
"""

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace

from app.observability.telemetry import FORBIDDEN_KEY_PARTS

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

LOGGER_NAME = "policy_rag"
_REDACTIONS = (
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer [REDACTED]"),
    (re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{12,}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "[REDACTED_API_KEY]"),
    (
        re.compile(r"(?i)\b(api[_-]?key|password|passwd|secret|token)\s*[=:]\s*\S+"),
        r"\1=[REDACTED]",
    ),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*"), "[REDACTED_JWT]"),
)
_MAX_VALUE_CHARS = 300


def redact(value: str) -> str:
    for pattern, replacement in _REDACTIONS:
        value = pattern.sub(replacement, value)
    return value


def _safe_fields(fields: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in fields.items():
        if any(part in key.lower() for part in FORBIDDEN_KEY_PARTS) or value is None:
            continue
        if isinstance(value, bool | int | float):
            safe[key] = value
        else:
            safe[key] = redact(str(value))[:_MAX_VALUE_CHARS]
    return safe


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": redact(record.getMessage()),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            payload["trace_id"] = format(ctx.trace_id, "032x")
            payload["span_id"] = format(ctx.span_id, "016x")
        payload.update(_safe_fields(getattr(record, "fields", {})))
        if record.exc_info and record.exc_info[0] is not None:
            # The exception type only: messages can carry user input or secrets.
            payload["error_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = _safe_fields(getattr(record, "fields", {}))
        extras = " ".join(f"{k}={v}" for k, v in fields.items())
        request_id = request_id_var.get()
        prefix = f"[{request_id}] " if request_id else ""
        return f"{record.levelname} {prefix}{redact(record.getMessage())} {extras}".rstrip()


def configure_logging(log_format: str = "json", level: str = "INFO") -> logging.Logger:
    """Install the stderr handler on the `policy_rag` logger (idempotent)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False
    for handler in list(logger.handlers):
        if getattr(handler, "_policy_rag", False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if log_format == "json" else TextFormatter())
    handler._policy_rag = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logger


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one structured event. `fields` are filtered and redacted before they are written."""
    logging.getLogger(LOGGER_NAME).log(level, event, extra={"fields": fields})
