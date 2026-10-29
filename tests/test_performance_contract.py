import time

from app.answering.engine import AnswerEngine
from app.retrieval.search import KnowledgeSearch
from scripts.benchmark_runtime import benchmark_engine, benchmark_search


def rows(count=100):
    return [
        {
            "chunk_id": f"adeno-{i}", "source_id": "adenocarcinoma-package",
            "page_type": "treatment_package", "source_url": "https://example.com/package",
            "title": "Package", "text": f"Adenocarcinoma outpatient program duration and cost detail {i}.",
        }
        for i in range(count)
    ]


def test_warm_lexical_retrieval_contract_is_under_100ms():
    metrics = benchmark_search(
        KnowledgeSearch(rows()),
        ["adenocarcinoma duration", "outpatient cost", "package details"],
        iterations=5,
    )
    assert metrics["retrieval_p95_ms"] < 100


def test_benchmark_measures_delayed_generation_without_cancelling_it():
    class DelayedGenerator:
        def generate(self, question, evidence, package_key=None):
            time.sleep(0.02)
            return "Retrieved outpatient program information."

    engine = AnswerEngine(KnowledgeSearch(rows(5)), generator=DelayedGenerator())
    metrics = benchmark_engine(engine, ["How long is the adenocarcinoma outpatient program?"])
    assert metrics["generation_p50_ms"] >= 20
    assert metrics["total_p50_ms"] >= metrics["generation_p50_ms"]
    assert metrics["max_evidence_chunks"] <= 3
    assert metrics["max_evidence_chars"] <= 6000
