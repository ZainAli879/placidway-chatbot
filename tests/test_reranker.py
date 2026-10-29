from app.retrieval.reranker import FlashRankReranker
from app.retrieval.search import SearchResult


def test_reranker_uses_best_passage_score_for_each_parent_chunk():
    class FakeRanker:
        def rerank(self, request):
            rows = list(request.passages)
            return [
                {**row, "score": 10.0 if "Included in the Out-Patient" in row["text"] else 0.1}
                for row in rows
            ]

    pricing = SearchResult("price", "adrenal", "treatment_package", "u", "Adrenal", "Price table only.", 2)
    inclusions = SearchResult(
        "include", "adrenal", "treatment_package", "u", "Adrenal",
        ("Introductory text.\n\n" + "background " * 200 +
         "\n\nIncluded in the Out-Patient Program\nWeekly lab work and imaging."), 1,
    )
    ranked = FlashRankReranker(ranker=FakeRanker()).rerank(
        "What does the outpatient package include?", [pricing, inclusions], top_n=1
    )
    assert [item.chunk_id for item in ranked] == ["include"]
    assert ranked[0].score == 10.0


def test_reranker_returns_empty_without_calling_model():
    class ExplodingRanker:
        def rerank(self, request):
            raise AssertionError("empty candidates must not call ONNX")

    assert FlashRankReranker(ranker=ExplodingRanker()).rerank("unknown", [], top_n=3) == []
