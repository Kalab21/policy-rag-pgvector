"""Ingest the policy documents into PostgreSQL.

    python -m scripts.ingest [--path sample_data/policies]

Safe to run repeatedly: unchanged documents are skipped without being re-embedded.
"""

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.db.pool import create_pool
from app.db.schema import init_schema
from app.embeddings.factory import get_embedding_provider
from app.ingestion.loader import load_documents
from app.ingestion.pipeline import ingest_documents


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--path", default=settings.sample_data_dir, help="directory of *.md files")
    args = parser.parse_args()

    init_schema(
        settings.database_url,
        settings.embedding_dim,
        settings.hnsw_m,
        settings.hnsw_ef_construction,
    )
    provider = get_embedding_provider(settings)
    documents = load_documents(Path(args.path))
    pool = create_pool(settings.database_url, 1, 2)
    try:
        results = ingest_documents(
            pool, provider, documents, settings.chunk_size, settings.chunk_overlap
        )
    finally:
        pool.close()

    print(f"model: {provider.model_name} ({provider.dimension} dimensions)")
    for r in results:
        print(f"  {r.action:<8} {r.name} v{r.version}: {r.chunks} chunks")
    total = sum(r.chunks for r in results)
    print(f"{len(results)} documents, {total} chunks")


if __name__ == "__main__":
    main()
