from app.answering.engine import AnswerEngine
from app.retrieval.search import SearchResult


PACKAGE_URL = (
    "https://www.placidway.com/package/7901/"
    "Alternative-Anal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC"
)


class ListSearch:
    def __init__(self):
        self.calls = []
        self.rows = [
            SearchResult(
                chunk_id=f"anal-{i}",
                source_id="anal-cancer-package",
                page_type="treatment_package",
                source_url=PACKAGE_URL,
                title="Alternative Anal Cancer Treatment Package",
                text=(
                    f"Anal cancer optional procedures list section {i}. "
                    f"Optional procedure detail and additional cost price evidence {i}."
                ),
                score=1.0 / i,
            )
            for i in range(1, 10)
        ]

    def search(self, query, *, top_k=5):
        self.calls.append(query)
        return self.rows[:top_k]


class StubGenerator:
    def generate(self, question, evidence, package_key=None):
        return f"Evidence chunks received: {len(evidence)}"


class StubReranker:
    def __init__(self):
        self.questions = []

    def rerank(self, question, candidates, *, top_n):
        self.questions.append(question)
        return list(candidates)[:top_n]


def test_list_question_uses_one_search_and_at_most_five_same_package_chunks():
    search = ListSearch()
    engine = AnswerEngine(search, generator=StubGenerator(), reranker=StubReranker())

    result = engine.answer(
        "What are the optional procedures for anal cancer?",
        intent_question="What are the optional procedures for anal cancer?",
    )

    assert len(search.calls) == 1
    assert len(result.evidence) == 4
    assert all(item.source_id == "anal-cancer-package" for item in result.evidence)


def test_ordinal_followup_keeps_compact_same_package_evidence():
    search = ListSearch()
    reranker = StubReranker()
    engine = AnswerEngine(search, generator=StubGenerator(), reranker=reranker)
    expanded = (
        "Follow-up about the same topic as this previous question: "
        "What are the optional procedures for anal cancer. "
        "The previous assistant answer was: LAK cells; Specific Killer Cell Antibody; "
        "Ethanol Tumor Ablation; MCA; Intratumoral Immunotherapy. "
        "Resolve references such as 'the second one' against that answer's list. "
        "The user now asks for price or cost: How much is the second one."
    )

    result = engine.answer(expanded, intent_question="How much is the second one?")

    assert len(search.calls) == 1
    assert 1 <= len(result.evidence) <= 4
    assert "Specific Killer Cell Antibody" in reranker.questions[0]


def test_focused_question_sends_top_three_reranked_chunks():
    search = ListSearch()
    engine = AnswerEngine(search, generator=StubGenerator(), reranker=StubReranker())
    result = engine.answer("What is the anal cancer outpatient price?")
    assert len(result.evidence) == 3


def test_equivalent_inclusion_wording_sends_four_chunks():
    search = ListSearch()
    engine = AnswerEngine(search, generator=StubGenerator(), reranker=StubReranker())
    result = engine.answer("What does the anal cancer outpatient package include?")
    assert len(result.evidence) == 4


def test_explicit_comparison_sends_five_reranked_chunks():
    search = ListSearch()
    engine = AnswerEngine(search, generator=StubGenerator(), reranker=StubReranker())
    result = engine.answer("Compare the differences across all anal cancer procedures and prices.")
    assert len(result.evidence) == 5


def test_numbered_multipart_question_sends_five_reranked_chunks():
    search = ListSearch()
    engine = AnswerEngine(search, generator=StubGenerator(), reranker=StubReranker())
    result = engine.answer(
        "1. What is the anal cancer outpatient price? "
        "2. How long does the anal cancer package last? "
        "3. What does the anal cancer package include?"
    )
    assert len(result.evidence) == 5


def test_dense_only_candidate_reaches_reranker_without_lexical_overlap():
    class DenseSearch:
        def search(self, query, *, top_k=5):
            return [SearchResult(
                chunk_id="semantic-only",
                source_id="clinic-profile",
                page_type="clinic_profile",
                source_url="https://example.com/clinic",
                title="Related care",
                text="Conceptually relevant evidence phrased with entirely different vocabulary.",
                score=1.0,
                retrieval_sources=("vector",),
            )]

    engine = AnswerEngine(DenseSearch(), generator=StubGenerator(), reranker=StubReranker())
    result = engine.answer("How does the center support immune resilience?")

    assert result.reason == "grounded_llm_answer"
    assert [item.chunk_id for item in result.evidence] == ["semantic-only"]
