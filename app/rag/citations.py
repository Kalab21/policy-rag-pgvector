"""Citation checking: every claim in an answer must point at a chunk that was supplied."""

import re
from dataclasses import dataclass

_MARKER = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class CitationCheck:
    answer: str  # with markers that point nowhere removed
    cited: list[int]  # valid 1-based source numbers, in order of first use
    invalid: list[int]  # markers that did not match a supplied source

    @property
    def grounded(self) -> bool:
        return bool(self.cited)


def validate_citations(answer: str, source_count: int) -> CitationCheck:
    cited: list[int] = []
    invalid: list[int] = []

    def keep(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if 1 <= number <= source_count:
            if number not in cited:
                cited.append(number)
            return match.group(0)
        if number not in invalid:
            invalid.append(number)
        return ""

    cleaned = re.sub(r"[ \t]{2,}", " ", _MARKER.sub(keep, answer)).strip()
    return CitationCheck(cleaned, cited, invalid)
