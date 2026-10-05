"""MCP tool inputs and the filter boundary, without a database or the protocol layer."""

import pytest
from pydantic import TypeAdapter, ValidationError

from app.mcp_server.tools import (
    MAX_TOP_K,
    DocumentName,
    PolicyTools,
    QueryText,
    ToolInputError,
    TopK,
    VersionText,
    merge_filters,
)
from app.models.schemas import SearchFilters
from tests.unit.test_rag_units import chunk


def validate(annotation: object, value: object) -> object:
    return TypeAdapter(annotation).validate_python(value)


# --- argument bounds ---------------------------------------------------------------------------


def test_query_text_is_stripped_and_bounded() -> None:
    assert validate(QueryText, "  late fee  ") == "late fee"
    for bad in ("", "   ", "x" * 1001):
        with pytest.raises(ValidationError):
            validate(QueryText, bad)


def test_top_k_is_bounded_tighter_than_the_http_api() -> None:
    assert MAX_TOP_K == 10
    assert validate(TopK, 1) == 1
    assert validate(TopK, 10) == 10
    for bad in (0, -1, 11, 1000):
        with pytest.raises(ValidationError):
            validate(TopK, bad)


@pytest.mark.parametrize("good", ["fee-schedule", "underwriting-policy", "a", "doc-2"])
def test_document_names_accept_the_stored_slug_format(good: str) -> None:
    assert validate(DocumentName, good) == good


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "../../etc/passwd",
        "/etc/passwd",
        "C:\\Windows\\system32",
        "fee schedule",
        "Fee-Schedule",
        "a;b",
        "x'; DROP TABLE documents; --",
        "-leading-dash",
        "a" * 81,
        "fee-schedule\n",
    ],
)
def test_document_names_reject_paths_sql_and_odd_characters(bad: str) -> None:
    with pytest.raises(ValidationError):
        validate(DocumentName, bad)


@pytest.mark.parametrize("good", ["1", "2.0", "3.1", "1.2.3"])
def test_versions_accept_dotted_numbers(good: str) -> None:
    assert validate(VersionText, good) == good


@pytest.mark.parametrize("bad", ["", "v2", "2.0-beta", "2.", "1.2.3.4", "../1", "1; DROP"])
def test_versions_reject_everything_else(bad: str) -> None:
    with pytest.raises(ValidationError):
        validate(VersionText, bad)


def test_filter_values_are_length_bounded() -> None:
    with pytest.raises(ValidationError):
        SearchFilters(category="x" * 201)
    with pytest.raises(ValidationError):
        SearchFilters(status="draft")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        SearchFilters(colour="red")  # type: ignore[call-arg]


# --- the filter boundary -----------------------------------------------------------------------


def test_caller_filters_are_kept_and_enforced_filters_are_added() -> None:
    assert merge_filters({"category": "fees"}, {"status": "current"}) == {
        "category": "fees",
        "status": "current",
    }
    assert merge_filters({}, {}) == {}


def test_a_caller_may_repeat_an_enforced_value() -> None:
    assert merge_filters({"status": "current"}, {"status": "current"}) == {"status": "current"}


def test_a_caller_cannot_change_an_enforced_filter() -> None:
    with pytest.raises(ToolInputError, match="fixed by the server"):
        merge_filters({"category": "privacy"}, {"category": "fees"})


class RecordingRetrieval:
    model_name = "stub"
    mode = "semantic"
    rerank_enabled = False

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict[str, str], str | None]] = []

    def search(self, query, top_k, filters, mode=None, **_kwargs):  # type: ignore[no-untyped-def]
        self.calls.append((query, top_k, dict(filters), mode))
        return [chunk(0.9, "text", 1)]


def test_enforced_filters_reach_retrieval_on_every_search() -> None:
    retrieval = RecordingRetrieval()
    tools = PolicyTools(retrieval, None, None, {"category": "fees"})  # type: ignore[arg-type]
    tools.search_policy("late fee", 3, SearchFilters(status="current"), "hybrid")
    assert retrieval.calls == [("late fee", 3, {"status": "current", "category": "fees"}, "hybrid")]


def test_a_search_that_tries_to_override_an_enforced_filter_never_reaches_retrieval() -> None:
    retrieval = RecordingRetrieval()
    tools = PolicyTools(retrieval, None, None, {"category": "fees"})  # type: ignore[arg-type]
    with pytest.raises(ToolInputError):
        tools.search_policy("anything", 3, SearchFilters(category="privacy"))
    assert retrieval.calls == []
