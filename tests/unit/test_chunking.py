from itertools import pairwise

import pytest

from app.ingestion.chunking import chunk_document, pack, split_sections
from app.ingestion.loader import parse_document
from app.models.domain import ParsedDocument

DOC = """---
name: demo
title: Demo Policy
version: "1.0"
category: demo
status: current
---

# Demo Policy

## Short section
One short fact. Another short fact.

## Long section
""" + " ".join(f"Sentence number {i} states a distinct rule about item {i}." for i in range(1, 41))


def _doc() -> ParsedDocument:
    return parse_document(DOC, "demo.md")


def test_sections_are_split_at_headings_and_the_title_is_not_a_section() -> None:
    sections = split_sections(_doc().body)
    assert [name for name, _ in sections] == ["Short section", "Long section"]


def test_text_before_the_first_section_is_kept_as_overview() -> None:
    sections = split_sections("Intro text.\n\n## Part\nBody.")
    assert [name for name, _ in sections] == ["Overview", "Part"]


def test_a_short_section_is_a_single_chunk() -> None:
    chunks = [c for c in chunk_document(_doc(), 700, 120) if c.section == "Short section"]
    assert len(chunks) == 1
    assert chunks[0].text == "One short fact. Another short fact."


def test_no_chunk_exceeds_the_size_and_none_crosses_a_section() -> None:
    chunks = chunk_document(_doc(), chunk_size=300, chunk_overlap=60)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 for c in chunks)
    long_chunks = [c for c in chunks if c.section == "Long section"]
    assert len(long_chunks) > 1
    assert all("Short" not in c.text for c in long_chunks)


def test_chunk_indexes_are_sequential_and_metadata_is_attached() -> None:
    chunks = chunk_document(_doc(), 300, 60)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    first = chunks[0]
    assert first.metadata == {
        "document": "demo",
        "title": "Demo Policy",
        "version": "1.0",
        "category": "demo",
        "status": "current",
        "section": "Short section",
    }


def test_embedded_text_carries_title_and_section_but_stored_text_does_not() -> None:
    chunk = chunk_document(_doc(), 700, 120)[0]
    assert chunk.embed_text.startswith("Demo Policy | Short section\n")
    assert chunk.text == "One short fact. Another short fact."


def test_consecutive_chunks_overlap() -> None:
    text = " ".join(f"Rule {i} applies to case {i}." for i in range(1, 30))
    chunks = pack(text, size=120, overlap=40)
    assert len(chunks) > 2
    for previous, following in pairwise(chunks):
        last_sentence = previous.rsplit(". ", 1)[-1]
        assert last_sentence in following


def test_every_sentence_survives_chunking() -> None:
    sentences = [f"Fact {i} is recorded here." for i in range(1, 25)]
    chunks = pack(" ".join(sentences), size=100, overlap=30)
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)


def test_a_sentence_longer_than_the_chunk_is_split_on_words() -> None:
    chunks = pack("word " * 100, size=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_chunking_is_deterministic() -> None:
    assert chunk_document(_doc(), 300, 60) == chunk_document(_doc(), 300, 60)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_size_or_overlap_is_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="chunk"):
        pack("A sentence.", size, overlap)
