from app.answering.engine import AnswerEngine
from app.retrieval.search import SearchResult


class StubSearch:
    def __init__(self):
        self.calls = []

    def search(self, query, *, top_k=5):
        self.calls.append(query)
        return [
            SearchResult(
                chunk_id="anal-1",
                source_id="anal-cancer-package",
                page_type="treatment_package",
                source_url="https://www.placidway.com/package/7901/Alternative-Anal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
                title="Alternative Anal Cancer Treatment Package",
                text="The anal cancer package page lists program prices and optional procedures.",
                score=1.0,
            )
        ]


class StubGenerator:
    def generate(self, question, evidence, package_key=None):
        return "The package page lists the relevant information."


def test_context_disclaimer_does_not_trigger_medical_advice_guard():
    search = StubSearch()
    engine = AnswerEngine(search, generator=StubGenerator())
    expanded = (
        "Follow-up about the same topic as this previous question: "
        "What are the optional procedures for anal cancer. "
        "The previous assistant answer was: This is website information, not medical advice. "
        "The user now asks for price or cost: How much is the second one."
    )

    result = engine.answer(expanded, intent_question="How much is the second one?")

    assert result.reason == "grounded_llm_answer"
    assert search.calls


def test_current_personal_medical_safety_question_still_triggers_guard():
    engine = AnswerEngine(StubSearch(), generator=StubGenerator())

    result = engine.answer(
        "Follow-up context mentions the anal cancer package and its previous answer.",
        intent_question="Is this treatment safe with my diabetes?",
    )

    assert result.abstained
    assert result.reason == "medical_safety_not_in_sources"


def test_current_weather_question_is_not_hidden_by_old_package_context():
    engine = AnswerEngine(StubSearch(), generator=StubGenerator())

    result = engine.answer(
        "Follow-up context mentions the anal cancer package.",
        intent_question="What is the weather in Cancun?",
    )

    assert result.abstained
    assert result.reason == "out_of_scope_weather"
