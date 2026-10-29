"""Small, deterministic retrieval metrics for regression checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from app.retrieval.search import SearchResult


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_chunk_ids: frozenset[str]


class Searcher(Protocol):
    def search(self, query: str, *, top_k: int) -> list[SearchResult]: ...


def evaluate_retrieval(
    searcher: Searcher,
    cases: Iterable[RetrievalCase],
    *,
    top_k: int = 5,
) -> dict[str, int | float]:
    """Return hit-rate recall and reciprocal-rank metrics at ``top_k``."""
    case_list = list(cases)
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    recall_total = 0.0
    reciprocal_rank = 0.0
    for case in case_list:
        results = searcher.search(case.query, top_k=top_k)
        retrieved_ids = {item.chunk_id for item in results}
        if case.relevant_chunk_ids:
            recall_total += len(retrieved_ids & case.relevant_chunk_ids) / len(case.relevant_chunk_ids)
        first_rank = next(
            (rank for rank, item in enumerate(results, start=1)
             if item.chunk_id in case.relevant_chunk_ids),
            None,
        )
        if first_rank is not None:
            reciprocal_rank += 1 / first_rank

    count = len(case_list)
    return {
        "case_count": count,
        "top_k": top_k,
        "recall_at_k": round(recall_total / count, 4) if count else 0.0,
        "mrr_at_k": round(reciprocal_rank / count, 4) if count else 0.0,
    }
