from pathlib import Path

import pytest

from app.ingestion.loader import DocumentFormatError, load_documents, parse_document
from app.ingestion.normalize import normalize_text

VALID = (
    "---\n"
    "name: a\n"
    "title: A Title\n"
    'version: "2.0"\n'
    "category: fees\n"
    "status: current\n"
    "---\n"
    "\n"
    "# A\n"
    "\n"
    "## S\n"
    "Text.\n"
)


def test_front_matter_is_parsed() -> None:
    doc = parse_document(VALID, "a.md")
    assert (doc.name, doc.title, doc.version, doc.category, doc.status) == (
        "a",
        "A Title",
        "2.0",
        "fees",
        "current",
    )
    assert doc.source_path == "a.md"
    assert "## S" in doc.body
    assert "name:" not in doc.body


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("# no front matter", "missing front matter"),
        ("---\nname: a\n# never closed", "not closed"),
        ("---\nname: a\n---\nbody", "missing front matter field"),
        (VALID.replace("current", "draft"), "status must be one of"),
        (VALID.replace("category: fees", "category fees"), "invalid front matter line"),
    ],
)
def test_malformed_documents_are_rejected(raw: str, message: str) -> None:
    with pytest.raises(DocumentFormatError, match=message):
        parse_document(raw, "bad.md")


def test_load_documents_reads_every_markdown_file_in_order(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text(VALID.replace("name: a", "name: b"), encoding="utf-8")
    (tmp_path / "a.md").write_text(VALID, encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("not a document", encoding="utf-8")
    assert [d.name for d in load_documents(tmp_path)] == ["a", "b"]


def test_an_empty_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path)


def test_the_bundled_sample_policies_all_parse() -> None:
    docs = load_documents(Path("sample_data/policies"))
    assert len(docs) == 7
    assert {d.status for d in docs} == {"current", "superseded"}
    assert len({(d.name, d.version) for d in docs}) == len(docs)


def test_normalization_folds_equivalent_text() -> None:
    assert normalize_text("A B\r\n\r\n\r\n\r\nC  \t D") == "A B\n\nC D"
    assert normalize_text("ﬁne") == "fine"  # the "fi" ligature is folded by NFKC
