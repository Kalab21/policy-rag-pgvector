"""One real Bedrock request, to confirm the adapter works against your AWS account.

    BEDROCK_MODEL_ID=<a model your account can invoke> BEDROCK_REGION=<region> \\
        python -m scripts.bedrock_smoke

Uses the standard AWS credential chain (environment, profile, or IAM role). It sends one
small question with two made-up source passages and checks that the structured, cited answer
comes back and validates. Nothing is stored. It makes a billable Bedrock call.

Exit codes: 0 = the live request succeeded; 2 = it could not be completed (missing
credentials, model access, or invalid model output). The unit tests mock the SDK, so this
script is the only live check.
"""

import sys

from app.core.config import get_settings
from app.models.domain import RetrievedChunk
from app.rag.factory import get_generator
from app.rag.generator import GenerationError


def _chunk(chunk_id: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        document="demo",
        title="Demo policy",
        version="1.0",
        category="demo",
        status="current",
        section="Fees",
        text=text,
        distance=0.1,
        similarity=0.9,
        metadata={},
    )


def main() -> int:
    settings = get_settings().model_copy(update={"llm_provider": "bedrock"})
    if not settings.bedrock_model_id:
        print("BLOCKED: set BEDROCK_MODEL_ID to a model your account can invoke.")
        return 2
    generator = get_generator(settings)
    chunks = [
        _chunk(1, "The late fee is the lesser of $35 or 5 percent of the unpaid installment."),
        _chunk(2, "Electronic statements are free. Paper statements cost $3 per month."),
    ]
    try:
        answer = generator.generate("What is the late fee?", chunks)
    except GenerationError as exc:
        print(f"NOT VALIDATED: {type(exc).__name__}: {exc}")
        return 2
    print(f"model: {generator.name}")
    print(f"answer: {answer}")
    if "[1]" not in answer:
        print("NOT VALIDATED: the answer did not cite source [1].")
        return 2
    print("LIVE BEDROCK REQUEST OK: structured, cited answer received and validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
