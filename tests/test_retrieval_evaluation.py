from app.retrieval.evaluation import RetrievalCase, evaluate_retrieval
from app.retrieval.search import SearchResult


def _result(chunk_id: str) -> SearchResult:
    return SearchResult(chunk_id, "source", "page", "https://example.com", "Title", "Text", 1.0)


class FakeSearcher:
    def search(self, query: str, *, top_k: int):
        rows = {
            "found first": ["expected-a", "other"],
            "found second": ["other", "expected-b"],
            "missed": ["other"],
        }
        return [_result(chunk_id) for chunk_id in rows[query]][:top_k]


def test_evaluate_retrieval_reports_recall_and_mrr():
    cases = [
        RetrievalCase("found first", frozenset({"expected-a"})),
        RetrievalCase("found second", frozenset({"expected-b"})),
        RetrievalCase("missed", frozenset({"expected-c"})),
    ]

    metrics = evaluate_retrieval(FakeSearcher(), cases, top_k=3)

    assert metrics == {
        "case_count": 3,
        "top_k": 3,
        "recall_at_k": 0.6667,
        "mrr_at_k": 0.5,
    }


def test_evaluate_retrieval_handles_no_cases():
    assert evaluate_retrieval(FakeSearcher(), [], top_k=5) == {
        "case_count": 0,
        "top_k": 5,
        "recall_at_k": 0.0,
        "mrr_at_k": 0.0,
    }


def test_recall_counts_each_relevant_chunk_not_only_any_hit():
    cases = [RetrievalCase("found first", frozenset({"expected-a", "missing"}))]

    metrics = evaluate_retrieval(FakeSearcher(), cases, top_k=3)

    assert metrics["recall_at_k"] == 0.5
