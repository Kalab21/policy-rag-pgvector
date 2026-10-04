"""Plain data types shared by ingestion and retrieval."""

from dataclasses import dataclass, field
from typing import Literal

DocumentStatus = Literal["current", "superseded"]


@dataclass(frozen=True)
class ParsedDocument:
    """A policy document read from disk, with its front matter."""

    name: str
    title: str
    version: str
    category: str
    status: DocumentStatus
    source_path: str
    body: str  # normalized markdown, without the front matter


@dataclass(frozen=True)
class Chunk:
    """A passage of a document, ready to be embedded and stored."""

    index: int
    section: str
    text: str  # what is shown to users and cited
    embed_text: str  # what is embedded: the text prefixed with its title and section
    text_hash: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestResult:
    name: str
    version: str
    action: Literal["created", "updated", "skipped"]
    chunks: int
