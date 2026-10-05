"""Quality floors for retrieval and the evidence gate, measured with the real models.

Both question sets are run: `tuning` (used while building the system) and `held_out`
(never used to choose a setting). The floors sit below the measured values reported in the
README so ordinary noise does not fail the build, while a real regression in chunking,
embedding, retrieval, reranking or the gate does. They are regression guards, not quality
claims: the sets are small and the corpus is tiny.
"""

from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.db.pool import create_pool
from app.embeddings.fastembed_provider import FastEmbedProvider
from app.evaluation.gold import load_gold
from app.evaluation.runner import run_evaluation
from app.ingestion.loader import load_documents
from app.ingestion.pipeline import ingest_documents
from app.rag.generator import ExtractiveGenerator
from app.rag.service import RagService
from app.retrieval.rerank import CrossEncoderReranker
from app.retrieval.service import RetrievalService
from tests.conftest import TEST_DIM

pytestmark = [pytest.mark.integration, pytest.mark.model]

# (configuration, metric, floor). "mrr" is a top-level metric; the others are "<name>@<k>".
RETRIEVAL_FLOORS = {
    "tuning": [
        ("semantic", "hit_at_k@5", 0.90),
        ("semantic", "mrr", 0.85),
        ("lexical", "mrr", 0.80),
        ("hybrid", "mrr", 0.80),
        ("semantic+rerank", "mrr", 0.90),
    ],
    "held_out": [
        ("semantic", "hit_at_k@5", 0.90),
        ("semantic", "mrr", 0.85),
        ("semantic", "ndcg_at_k@5", 0.85),
        ("lexical", "mrr", 0.90),
        ("hybrid", "mrr", 0.90),
        ("semantic+rerank", "mrr", 0.90),
        ("hybrid+rerank", "mrr", 0.90),
    ],
}
ASK_FLOORS = {
    "tuning": {"refusal_rate": 0.80, "correct_rate": 0.55},
    "held_out": {"refusal_rate": 0.75, "correct_rate": 0.50},
}


@pytest.fixture(scope="module")
def reports(database_url: str) -> dict[str, dict[str, Any]]:
    import psycopg

    from app.db.schema import init_schema

    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, documents CASCADE")
    init_schema(database_url, TEST_DIM)
    settings = Settings(database_url=database_url, embedding_dim=TEST_DIM, _env_file=None)  # type: ignore[call-arg]
    embedder = FastEmbedProvider(settings.embedding_model)
    pool = create_pool(database_url)
    try:
        docs = load_documents(Path("sample_data/policies"))
        ingest_documents(pool, embedder, docs, settings.chunk_size, settings.chunk_overlap)
        retrieval = RetrievalService(
            pool, embedder, settings.hnsw_ef_search, True, reranker=CrossEncoderReranker()
        )
        rag = RagService(
            retrieval,
            ExtractiveGenerator(embedder),
            settings.evidence_min_similarity,
            settings.rag_max_context_chunks,
        )
        return {
            name: run_evaluation(retrieval, rag, load_gold(Path(f"eval/{name}.json")))
            for name in ("tuning", "held_out")
        }
    finally:
        pool.close()


@pytest.mark.parametrize(
    ("dataset", "configuration", "metric", "floor"),
    [(d, c, m, f) for d, rows in RETRIEVAL_FLOORS.items() for c, m, f in rows],
)
def test_retrieval_meets_its_floor(
    reports: dict[str, dict[str, Any]], dataset: str, configuration: str, metric: str, floor: float
) -> None:
    result = reports[dataset]["retrieval_modes"][configuration]
    if "@" in metric:
        name, k = metric.split("@")
        value = result[name][k]
    else:
        value = result[metric]
    assert value >= floor, f"{dataset}/{configuration} {metric} = {value:.3f} < {floor}"


@pytest.mark.parametrize("dataset", ["tuning", "held_out"])
def test_answer_and_refusal_behaviour_meets_its_floor(
    reports: dict[str, dict[str, Any]], dataset: str
) -> None:
    ask = reports[dataset]["ask"]
    floors = ASK_FLOORS[dataset]
    assert ask["unanswerable"]["refusal_rate"] >= floors["refusal_rate"]
    assert ask["answerable"]["correct_rate"] >= floors["correct_rate"]


def test_reranking_costs_latency_that_the_report_records(
    reports: dict[str, dict[str, Any]],
) -> None:
    """The default stays reranking-off partly because of this cost, so keep measuring it."""
    latency = reports["tuning"]["latency"]
    assert latency["semantic+rerank"]["p50_ms"] > latency["semantic"]["p50_ms"]
