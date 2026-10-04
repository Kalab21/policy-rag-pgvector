"""The gold question set used to evaluate retrieval and the evidence gate."""

import json
from dataclasses import dataclass, field
from pathlib import Path

ChunkKey = tuple[str, str, str]  # (document, version, section)


@dataclass(frozen=True)
class AnswerableQuestion:
    id: str
    question: str
    relevant: frozenset[ChunkKey]
    answer_contains: tuple[str, ...]
    filters: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class UnanswerableQuestion:
    id: str
    question: str
    kind: str  # out_of_domain | near_domain


@dataclass(frozen=True)
class GoldSet:
    answerable: list[AnswerableQuestion]
    unanswerable: list[UnanswerableQuestion]


def load_gold(path: Path) -> GoldSet:
    raw = json.loads(path.read_text(encoding="utf-8"))
    answerable = [
        AnswerableQuestion(
            id=item["id"],
            question=item["question"],
            relevant=frozenset(
                (r["document"], r["version"], r["section"]) for r in item["relevant"]
            ),
            answer_contains=tuple(item["answer_contains"]),
            filters=dict(item.get("filters", {})),
        )
        for item in raw["answerable"]
    ]
    unanswerable = [
        UnanswerableQuestion(item["id"], item["question"], item["kind"])
        for item in raw["unanswerable"]
    ]
    ids = [q.id for q in answerable] + [q.id for q in unanswerable]
    if len(set(ids)) != len(ids):
        raise ValueError("gold question ids must be unique")
    if not answerable or not unanswerable:
        raise ValueError("the gold set needs both answerable and unanswerable questions")
    return GoldSet(answerable, unanswerable)
