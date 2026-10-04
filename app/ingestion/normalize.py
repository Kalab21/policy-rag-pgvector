"""Text normalization applied before parsing, chunking and hashing."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Unicode-normalize, unify line endings and collapse stray whitespace.

    NFKC folds visually identical characters (non-breaking spaces, ligatures, full-width
    forms) so that the same passage always hashes and embeds the same way.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
