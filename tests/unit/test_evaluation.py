import json
from pathlib import Path

import pytest

from app.evaluation.gold import load_gold
from app.evaluation.metrics import hit_at_k, mean, recall_at_k, reciprocal_rank
from app.ingestion.chunking import split_sections
from app.ingestion.loader import load_documents
from app.rag.generator import ExtractiveGenerator
from tests.fakes import HashingEmbedder
from tests.unit.test_rag_units import chunk

RANKED = ["a", "b", "c", "d"]


def test_hit_at_k_is_one_when_a_relevant_item_is_in_the_top_k() -> None:
    assert hit_at_k(RANKED, {"c"}, 3) == 1.0
    assert hit_at_k(RANKED, {"c"}, 2) == 0.0
    assert hit_at_k(RANKED, {"z"}, 4) == 0.0


def test_recall_at_k_is_the_share_of_relevant_items_found() -> None:
    assert recall_at_k(RANKED, {"a", "d"}, 1) == 0.5
    assert recall_at_k(RANKED, {"a", "d"}, 4) == 1.0
    assert recall_at_k(RANKED, {"a", "z"}, 4) == 0.5


def test_recall_needs_something_relevant() -> None:
    with pytest.raises(ValueError, match="relevant"):
        recall_at_k(RANKED, set(), 3)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(RANKED, {"a"}) == 1.0
    assert reciprocal_rank(RANKED, {"c", "d"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"z"}) == 0.0


def test_mean_of_nothing_is_zero() -> None:
    assert mean([]) == 0.0
    assert mean([1.0, 0.0, 0.5]) == 0.5


def test_gold_file_loads_and_covers_answerable_unanswerable_and_superseded() -> None:
    gold = load_gold(Path("eval/gold.json"))
    assert len(gold.answerable) >= 30
    assert len(gold.unanswerable) >= 10
    assert {u.kind for u in gold.unanswerable} == {"out_of_domain", "near_domain"}
    assert any(q.filters.get("status") == "superseded" for q in gold.answerable)


def test_every_gold_target_is_a_real_section_of_a_sample_document() -> None:
    """Guards against a typo silently turning a question into an impossible one."""
    sections = {
        (doc.name, doc.version, name)
        for doc in load_documents(Path("sample_data/policies"))
        for name, _ in split_sections(doc.body)
    }
    for q in load_gold(Path("eval/gold.json")).answerable:
        for key in q.relevant:
            assert key in sections, f"{q.id}: {key} is not a section in sample_data"


def test_each_expected_fact_appears_in_its_relevant_section_text() -> None:
    texts = {
        (doc.name, doc.version, name): text.lower()
        for doc in load_documents(Path("sample_data/policies"))
        for name, text in split_sections(doc.body)
    }
    for q in load_gold(Path("eval/gold.json")).answerable:
        combined = " ".join(texts[key] for key in q.relevant)
        # The answer is quoted from the section, so the expected facts must exist there.
        assert all(fact.lower() in combined for fact in q.answer_contains), q.id


def test_gold_ids_must_be_unique(tmp_path: Path) -> None:
    gold = json.loads(Path("eval/gold.json").read_text(encoding="utf-8"))
    gold["answerable"].append(dict(gold["answerable"][0]))
    path = tmp_path / "dup.json"
    path.write_text(json.dumps(gold), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_gold(path)


def test_extractive_generator_can_rank_sentences_with_an_embedder() -> None:
    chunks = [chunk(0.9, "Statements are mailed monthly. Paper statements cost $3 per month.")]
    answer = ExtractiveGenerator(HashingEmbedder()).generate("paper statements cost", chunks)
    assert "Paper statements cost $3 per month. [1]" in answer
    assert "mailed monthly" not in answer
