"""Chunking strategy: split by section, then pack sentences up to a size with overlap.

Strategy
--------
1. A document is first split at its markdown headings, so a chunk never crosses a section
   boundary and always knows which section it belongs to.
2. A section that fits in `chunk_size` characters becomes one chunk.
3. A longer section is split into sentences (and list lines) and packed greedily into chunks
   of at most `chunk_size` characters. Each new chunk starts with the trailing sentences of
   the previous one, up to `chunk_overlap` characters, so a fact that sits on a boundary
   appears whole in at least one chunk.
4. A single sentence longer than `chunk_size` is split on word boundaries.

The text that is embedded is the chunk prefixed with the document title and section name,
because a passage such as "The fee is $25." is ambiguous without them. The text that is
stored, shown and cited is the chunk itself.
"""

import hashlib
import re

from app.models.domain import Chunk, ParsedDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 120


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_sections(body: str) -> list[tuple[str, str]]:
    """(heading, text) pairs. A top-level `#` title is the document title, not a section."""
    sections: list[tuple[str, str]] = []
    heading = "Overview"
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append((heading, text))
        buffer.clear()

    for line in body.split("\n"):
        match = _HEADING.match(line)
        if match is None:
            buffer.append(line)
            continue
        level = len(match.group(1))
        if level == 1:
            continue  # the document title
        flush()
        heading = match.group(2).strip()
    flush()
    return sections


def _units(text: str) -> list[str]:
    units: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            units.extend(part.strip() for part in _SENTENCE_END.split(line) if part.strip())
    return units


def _split_long(unit: str, size: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for word in unit.split(" "):
        if current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def pack(text: str, size: int, overlap: int) -> list[str]:
    """Greedy sentence packing with overlap. Pure function, easy to test."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("chunk overlap must be at least 0 and smaller than the chunk size")

    units: list[str] = []
    for unit in _units(text):
        units.extend(_split_long(unit, size) if len(unit) > size else [unit])

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    has_new = False  # does `current` hold anything beyond the overlap carried from the last chunk

    for unit in units:
        added = len(unit) + (1 if current else 0)
        if current and length + added > size:
            chunks.append(" ".join(current))
            tail: list[str] = []
            for earlier in reversed(current):
                candidate = [earlier, *tail]
                if len(" ".join(candidate)) > overlap:
                    break
                tail = candidate
            current = tail
            length = len(" ".join(current))
            has_new = False
            added = len(unit) + (1 if current else 0)
        current.append(unit)
        length += added
        has_new = True

    if current and has_new:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(
    doc: ParsedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, section_text in split_sections(doc.body):
        for text in pack(section_text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    index=len(chunks),
                    section=section,
                    text=text,
                    embed_text=f"{doc.title} | {section}\n{text}",
                    text_hash=sha256(text),
                    metadata={
                        "document": doc.name,
                        "title": doc.title,
                        "version": doc.version,
                        "category": doc.category,
                        "status": doc.status,
                        "section": section,
                    },
                )
            )
    return chunks
