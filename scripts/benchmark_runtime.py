"""Measure warm hybrid retrieval and optional end-to-end response performance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from app.answering.engine import AnswerEngine
from app.retrieval.hybrid_search import HybridKnowledgeSearch
from app.retrieval.reranker import FlashRankReranker


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def benchmark_search(searcher, queries: list[str], *, iterations: int = 10) -> dict:
    for query in queries:
        searcher.search(query, top_k=12)
    samples: list[float] = []
    for _ in range(iterations):
        for query in queries:
            started = time.perf_counter()
            searcher.search(query, top_k=12)
            samples.append((time.perf_counter() - started) * 1000)
    return {
        "retrieval_p50_ms": round(statistics.median(samples), 3),
        "retrieval_p95_ms": round(_percentile(samples, 0.95), 3),
    }


def benchmark_engine(engine: AnswerEngine, queries: list[str]) -> dict:
    answers = [engine.answer(query) for query in queries]
    generation = [answer.generation_ms for answer in answers]
    total = [answer.latency_ms for answer in answers]
    return {
        "generation_p50_ms": round(statistics.median(generation), 3),
        "total_p50_ms": round(statistics.median(total), 3),
        "max_evidence_chunks": max((len(answer.evidence) for answer in answers), default=0),
        "max_evidence_chars": max((sum(len(item.text) for item in answer.evidence) for answer in answers), default=0),
        "warm_target_seconds": 4,
        "target_met": max(total, default=0) <= 4000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--chroma", type=Path, default=Path("data/processed/chroma"))
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    searcher = HybridKnowledgeSearch.from_jsonl(
        args.index, persist_directory=args.chroma
    )
    queries = [
        "How long is the adenocarcinoma outpatient program?",
        "What is included in the adrenal cancer package?",
        "What information is available about the ITC clinic?",
    ]
    metrics = benchmark_search(searcher, queries, iterations=args.iterations)
    metrics.update(benchmark_engine(AnswerEngine(searcher, reranker=FlashRankReranker()), queries))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
