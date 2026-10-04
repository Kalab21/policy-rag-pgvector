"""The evidence gate: decide whether retrieval found enough support to answer at all.

The decision is made on measured vector similarity, before any text is generated. A
question whose best chunk is not similar enough to it is refused rather than answered
from loosely related text.
"""

from dataclasses import dataclass
from typing import Literal

from app.models.domain import RetrievedChunk

EvidenceStatus = Literal["sufficient", "insufficient"]


@dataclass(frozen=True)
class EvidenceAssessment:
    status: EvidenceStatus
    reason: str
    best_similarity: float | None
    threshold: float
    chunks_considered: int
    chunks_used: int


def assess_evidence(
    chunks: list[RetrievedChunk],
    min_similarity: float,
    max_context_chunks: int,
) -> tuple[EvidenceAssessment, list[RetrievedChunk]]:
    """Returns the assessment and the chunks that may be used as context.

    Chunks are ranked by similarity already; those below `min_similarity` are dropped and
    at most `max_context_chunks` of the rest are kept.
    """
    if not chunks:
        return (
            EvidenceAssessment("insufficient", "no_chunks_retrieved", None, min_similarity, 0, 0),
            [],
        )
    best = max(c.similarity for c in chunks)
    usable = [c for c in chunks if c.similarity >= min_similarity][:max_context_chunks]
    if not usable:
        return (
            EvidenceAssessment(
                "insufficient",
                f"best similarity {best:.3f} is below the threshold {min_similarity:.2f}",
                best,
                min_similarity,
                len(chunks),
                0,
            ),
            [],
        )
    return (
        EvidenceAssessment(
            "sufficient",
            f"{len(usable)} chunk(s) at or above the threshold {min_similarity:.2f}",
            best,
            min_similarity,
            len(chunks),
            len(usable),
        ),
        usable,
    )
