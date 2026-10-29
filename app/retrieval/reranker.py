"""Lightweight semantic reranking for the compact BM25 candidate pool."""
from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Sequence

from app.retrieval.search import SearchResult


class FlashRankReranker:
    def __init__(self, ranker=None) -> None:
        if ranker is None:
            from flashrank import Ranker
            cache_dir = Path(tempfile.gettempdir()) / "placidway-flashrank"
            cache_dir.mkdir(parents=True, exist_ok=True)
            ranker = Ranker(
                model_name="ms-marco-TinyBERT-L-2-v2",
                cache_dir=str(cache_dir),
                max_length=512,
                log_level="WARNING",
            )
        self.ranker = ranker

    def rerank(
        self, question: str, candidates: Sequence[SearchResult], *, top_n: int
    ) -> list[SearchResult]:
        if not candidates or top_n < 1:
            return []
        from flashrank import RerankRequest
        by_id = {item.chunk_id: item for item in candidates}
        passages = []
        for item in candidates:
            text = item.text
            starts = range(0, max(1, len(text)), 800)
            for index, start in enumerate(starts):
                passage = text[start:start + 1200].strip()
                if passage:
                    passages.append({
                        "id": f"{item.chunk_id}::{index}",
                        "text": f"{item.title}\n{passage}",
                    })
        request = RerankRequest(
            query=question,
            passages=passages,
        )
        ranked = self.ranker.rerank(request)
        best_scores: dict[str, float] = {}
        for item in ranked:
            parent_id = str(item["id"]).rsplit("::", 1)[0]
            best_scores[parent_id] = max(best_scores.get(parent_id, float("-inf")), float(item["score"]))
        parent_ids = sorted(best_scores, key=lambda chunk_id: (-best_scores[chunk_id], chunk_id))
        reranked: list[SearchResult] = []
        for chunk_id in parent_ids[:top_n]:
            item = by_id.get(chunk_id)
            if item is None:
                continue
            reranked.append(SearchResult(
                item.chunk_id,
                item.source_id,
                item.page_type,
                item.source_url,
                item.title,
                item.text,
                best_scores[chunk_id],
                item.retrieval_sources,
            ))
        return reranked
