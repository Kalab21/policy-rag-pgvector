"""Answer generation. The generator only ever sees chunks that passed the evidence gate.

Two implementations:
- ExtractiveGenerator (default): no model and no network. It selects the sentences of the
  retrieved chunks that best match the question and quotes them with [n] markers.
- OpenAICompatibleGenerator: calls a chat-completions endpoint (OpenAI, Ollama, vLLM, ...)
  that the operator configures. It is not used unless LLM_PROVIDER says so.
"""

import re
from typing import Protocol

import httpx

from app.models.domain import RetrievedChunk

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
        "there",
        "that",
        "this",
        "these",
        "those",
        "have",
        "has",
        "had",
        "any",
        "many",
        "much",
        "long",
    )
)


class AnswerGenerator(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """An answer that cites its sources as [1], [2], ... (positions in `chunks`)."""
        ...


def _stems(text: str) -> set[str]:
    # A crude 5-letter prefix stem is enough to match "retain"/"retention", "fee"/"fees".
    return {w[:5] for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line or line.startswith("#"):
            continue
        out.extend(s.strip() for s in _SENTENCE_END.split(line) if s.strip())
    return out


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_sentences: int = 3) -> None:
        self._max = max_sentences

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        wanted = _stems(question)
        scored: list[tuple[int, int, int, str]] = []  # (overlap, -chunk_pos, -sent_pos, text)
        for position, chunk in enumerate(chunks):
            for index, sentence in enumerate(_sentences(chunk.text)):
                overlap = len(wanted & _stems(sentence))
                scored.append((overlap, -position, -index, f"{sentence} [{position + 1}]"))
        if not scored:
            return ""
        scored.sort(reverse=True)
        best = scored[0][0]
        # Keep sentences that match the question about as well as the best one; if nothing
        # overlaps at all, quote the first sentence of the best-ranked chunk.
        floor = max(1, best - 1) if best else 0
        picked = [s for s in scored if s[0] >= floor][: self._max]
        picked.sort(key=lambda s: (-s[1], -s[2]))  # restore reading order
        return " ".join(s[3] for s in picked)


SYSTEM_PROMPT = (
    "You answer questions about company policy using ONLY the numbered sources provided. "
    "Cite every statement with its source number in square brackets, like [1]. "
    "If the sources do not contain the answer, reply exactly: INSUFFICIENT_EVIDENCE. "
    "Do not use outside knowledge."
)


class OpenAICompatibleGenerator:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )
        self._model = model

    @property
    def name(self) -> str:
        return f"openai-compatible:{self._model}"

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        sources = "\n\n".join(
            f"[{i}] ({c.title} v{c.version}, {c.section})\n{c.text}"
            for i, c in enumerate(chunks, start=1)
        )
        response = self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"},
                ],
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return str(content).strip()
