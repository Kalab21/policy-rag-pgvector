"""Quality floor for retrieval and the evidence gate, measured with the real model.

The floors sit below the measured values (see README) so ordinary noise does not fail the
build, but a real regression in chunking, embedding or the gate does.
"""

from pathlib import Path

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


def test_retrieval_and_gate_meet_the_quality_floor(clean_db: str) -> None:
    settings = Settings(database_url=clean_db, embedding_dim=TEST_DIM, _env_file=None)  # type: ignore[call-arg]
    embedder = FastEmbedProvider(settings.embedding_model)
    pool = create_pool(clean_db)
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
        report = run_evaluation(retrieval, rag, load_gold(Path("eval/gold.json")))
    finally:
        pool.close()

    current = report["retrieval_current_policy"]
    assert current["hit_at_k"]["5"] >= 0.9
    assert current["mrr"] >= 0.8
    ask = report["ask"]
    assert ask["unanswerable"]["refusal_rate"] >= 0.8
    assert ask["answerable"]["correct_rate"] >= 0.55
