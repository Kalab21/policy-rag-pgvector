"""Read policy documents (markdown with a small front-matter block) from a directory."""

from pathlib import Path

from app.ingestion.normalize import normalize_text
from app.models.domain import DocumentStatus, ParsedDocument

REQUIRED_FIELDS = ("name", "title", "version", "category", "status")
_STATUSES: tuple[DocumentStatus, ...] = ("current", "superseded")


class DocumentFormatError(ValueError):
    """A document is missing front matter or has an invalid field."""


def _parse_front_matter(raw: str, path: str) -> tuple[dict[str, str], str]:
    lines = raw.split("\n")
    if not lines or lines[0].strip() != "---":
        raise DocumentFormatError(f"{path}: missing front matter (file must start with '---')")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise DocumentFormatError(f"{path}: front matter is not closed with '---'") from None

    fields: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise DocumentFormatError(f"{path}: invalid front matter line {line!r}")
        fields[key.strip()] = value.strip().strip('"').strip("'")
    return fields, "\n".join(lines[end + 1 :])


def parse_document(raw: str, source_path: str) -> ParsedDocument:
    normalized = normalize_text(raw)
    fields, body = _parse_front_matter(normalized, source_path)

    missing = [name for name in REQUIRED_FIELDS if not fields.get(name)]
    if missing:
        raise DocumentFormatError(f"{source_path}: missing front matter field(s): {missing}")
    status = fields["status"]
    if status not in _STATUSES:
        raise DocumentFormatError(
            f"{source_path}: status must be one of {_STATUSES}, not {status!r}"
        )

    return ParsedDocument(
        name=fields["name"],
        title=fields["title"],
        version=fields["version"],
        category=fields["category"],
        status=status,
        source_path=source_path,
        body=body.strip(),
    )


def load_documents(directory: Path) -> list[ParsedDocument]:
    """Every *.md file under `directory`, in a stable order."""
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no markdown documents found in {directory}")
    return [parse_document(p.read_text(encoding="utf-8"), p.name) for p in paths]
