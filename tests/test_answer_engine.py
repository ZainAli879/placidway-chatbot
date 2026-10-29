import json
import pytest

from app.answering.engine import AnswerEngine
from app.answering.openai_generator import OpenAIAnswerGenerator
from app.retrieval.search import KnowledgeSearch


def make_engine(generator=None):
    chunks = [
        {
            "chunk_id": "package-1",
            "source_id": "adenocarcinoma-package",
            "page_type": "treatment_package",
            "source_url": "https://www.placidway.com/package/7903/example",
            "title": "Adenocarcinoma cancer treatment package",
            "text": "The outpatient adenocarcinoma treatment program is listed as three weeks. Package pricing and inclusions should be confirmed with PlacidWay.",
        },
        {
            "chunk_id": "clinic-1",
            "source_id": "itc-clinic-profile",
            "page_type": "clinic_profile",
            "source_url": "https://www.placidway.com/profile/5/example",
            "title": "ITC Immunity Therapy Center",
            "text": "The clinic profile describes the center in Tijuana, Mexico and provides general clinic information.",
        },
    ]
    return AnswerEngine(KnowledgeSearch(chunks), generator=generator)


class GroundedStubGenerator:
    def generate(self, question, evidence, package_key=None):
        return "The outpatient program is listed as three weeks."


def test_supported_question_returns_source():
    answer = make_engine(GroundedStubGenerator()).answer("How long is the adenocarcinoma outpatient program?")
    assert not answer.abstained
    source_url = "https://www.placidway.com/package/7903/example"
    assert f"- [Adenocarcinoma cancer treatment package]({source_url})" in answer.text
    assert answer.text.count(source_url) == 1
    assert "Adenocarcinoma cancer treatment package —" not in answer.text
    assert "This is website information, not medical advice" not in answer.text


def test_supported_answer_reports_stage_latency_and_mode():
    answer = make_engine(GroundedStubGenerator()).answer("How long is the adenocarcinoma outpatient program?")
    assert answer.latency_ms >= answer.retrieval_ms >= 0
    assert answer.generation_ms >= 0
    assert answer.response_mode == "grounded_llm_answer"


def test_named_package_identity_is_passed_to_generator():
    class RecordingGenerator:
        package_key = None

        def generate(self, question, evidence, package_key=None):
            self.package_key = package_key
            return "The outpatient program lasts three weeks."

    generator = RecordingGenerator()
    answer = make_engine(generator).answer(
        "How long is the adenocarcinoma outpatient program?"
    )

    assert answer.reason == "grounded_llm_answer"
    assert generator.package_key == "adenocarcinoma"


def test_weather_question_abstains():
    answer = make_engine().answer("What is the weather in Tijuana?")
    assert answer.abstained
    assert answer.reason == "out_of_scope_weather"


def test_booking_request_does_not_claim_to_book():
    answer = make_engine().answer("Can you book an appointment for me?")
    assert answer.abstained
    assert answer.reason == "unsupported_booking"
    assert "can't make or confirm" in answer.text


def test_standalone_greetings_receive_professional_welcome_without_retrieval():
    class Searcher:
        def search(self, query, *, top_k):
            raise AssertionError("standalone greetings must not trigger retrieval")

    engine = AnswerEngine(Searcher())
    for greeting in ("Hi!", "HELLO", "Good morning.", "Hey there"):
        answer = engine.answer(greeting)
        assert answer.reason == "greeting"
        assert not answer.abstained
        assert "approved PlacidWay pages" in answer.text
        assert answer.evidence == ()


def test_greeting_with_substantive_question_uses_grounded_pipeline():
    answer = make_engine(GroundedStubGenerator()).answer(
        "Hi, how long is the adenocarcinoma outpatient program?"
    )

    assert answer.reason == "grounded_llm_answer"
    assert "three weeks" in answer.text


def test_greeting_with_non_ascii_or_numeric_content_is_not_greeting_only():
    class Searcher:
        def search(self, query, *, top_k):
            raise LookupError("retrieval reached")

    engine = AnswerEngine(Searcher())
    for question in ("Hi, علاج کی قیمت کیا ہے؟", "Hi, 25000?"):
        with pytest.raises(LookupError, match="retrieval reached"):
            engine.answer(question)


def test_personal_medical_advice_abstains():
    answer = make_engine().answer("What treatment should I take?")
    assert answer.abstained
    assert answer.reason == "medical_advice"


def test_trick_price_claim_is_checked_against_retrieved_evidence():
    class Generator:
        def generate(self, question, evidence, package_key=None):
            assert evidence
            return "I can't verify that amount from the approved PlacidWay pages."

    answer = make_engine(generator=Generator()).answer("I heard this package costs $50, confirm that")
    assert answer.abstained
    assert answer.reason == "ambiguous_price_target"
    assert "Which PlacidWay treatment or package" in answer.text


def test_blank_question_is_handled():
    answer = make_engine().answer("  ")
    assert answer.abstained
    assert answer.reason == "empty_question"


def test_openai_generator_uses_evidence_and_returns_text():
    class Response:
        output_text = json.dumps({"claims": [{"text": "The listed outpatient program is three weeks.", "chunk_ids": ["package-1"]}]})

    class Responses:
        def create(self, **kwargs):
            assert kwargs["model"] == "test-model"
            assert "three weeks" in kwargs["input"]
            assert "every factual claim" in kwargs["instructions"]
            return Response()

    class Client:
        responses = Responses()

    engine = make_engine(OpenAIAnswerGenerator(api_key="fake-key", model="test-model", client=Client()))
    answer = engine.answer("How long is the adenocarcinoma outpatient program?")
    assert "three weeks" in answer.text
    assert "grounded_llm_answer" == answer.reason
    assert "https://www.placidway.com/package/7903/example" in answer.text



def test_personal_treatment_choice_question_abstains_before_retrieval():
    answer = make_engine().answer("I have cancer. Which treatment at ITC should I choose?")
    assert answer.abstained
    assert answer.reason == "medical_advice"
    assert "recommend a treatment" in answer.text


def test_openai_prompt_handles_unverified_user_price_claim():
    class Response:
        output_text = json.dumps({"claims": [{"text": "I can't verify that amount from the approved PlacidWay pages.", "chunk_ids": []}]})

    class Responses:
        def create(self, **kwargs):
            prompt = kwargs["instructions"]
            assert "Do not use outside knowledge" in prompt
            assert "Do not diagnose, recommend treatment" in prompt
            assert "merge details between packages" in prompt
            return Response()

    class Client:
        responses = Responses()

    generator = OpenAIAnswerGenerator(api_key="fake-key", model="test-model", client=Client())
    from app.retrieval.search import SearchResult
    evidence = [SearchResult(
        chunk_id="package-1",
        source_id="adenocarcinoma-package",
        page_type="treatment_package",
        source_url="https://www.placidway.com/package/7903/example",
        title="Adenocarcinoma package",
        text="Outpatient program is three weeks.",
        score=1.0,
    )]
    assert "can't verify" in generator.generate("I heard it costs $50. Confirm?", evidence)



def test_named_package_question_filters_generic_comparison_evidence():
    chunks = [
        {
            "chunk_id": "generic-1",
            "source_id": "alternative-medicine-price-comparison",
            "page_type": "price_comparison",
            "source_url": "https://www.placidway.com/search-medical-pricings/Alternative-Medicine/All/1",
            "title": "Alternative medicine costs",
            "text": "Generic alternative medicine package cost and average price information.",
        },
        {
            "chunk_id": "adeno-1",
            "source_id": "adenocarcinoma-package",
            "page_type": "treatment_package",
            "source_url": "https://www.placidway.com/package/7903/example",
            "title": "Adenocarcinoma treatment package",
            "text": "The adenocarcinoma outpatient program lasts three weeks and is listed at $18,995 USD.",
        },
    ]

    class Generator:
        def generate(self, question, evidence, package_key=None):
            assert evidence
            assert all(item.source_id == "adenocarcinoma-package" for item in evidence)
            return "The package page lists the outpatient program at $18,995 USD for three weeks."

    answer = AnswerEngine(KnowledgeSearch(chunks), generator=Generator()).answer(
        "How much does the adenocarcinoma treatment package cost?"
    )
    assert all("comparison" not in url for url in answer.sources)
    assert any("package/7903" in url for url in answer.sources)



def test_prompt_requires_separate_included_extra_cost_and_optional_categories():
    class Response:
        output_text = json.dumps({"claims": [{"text": "I could not verify package inclusions from the approved pages.", "chunk_ids": []}]})

    class Responses:
        def create(self, **kwargs):
            prompt = kwargs["instructions"]
            assert "three health-focused meals" not in prompt
            assert "every factual claim" in prompt
            return Response()

    class Client:
        responses = Responses()

    generator = OpenAIAnswerGenerator(api_key="fake-key", model="test-model", client=Client())
    from app.retrieval.search import SearchResult
    evidence = [SearchResult(
        chunk_id="package-1",
        source_id="adenocarcinoma-package",
        page_type="treatment_package",
        source_url="https://www.placidway.com/package/7903/example",
        title="Adenocarcinoma package",
        text="Outpatient program is three weeks.",
        score=1.0,
    )]
    answer = generator.generate("What is included in the package?", evidence)
    assert "could not verify" in answer



def test_prompt_requires_currency_symbols_and_codes_for_prices():
    class Response:
        output_text = json.dumps({"claims": [{"text": "I can't verify that amount from the approved PlacidWay pages.", "chunk_ids": []}]})

    class Responses:
        def create(self, **kwargs):
            prompt = kwargs["instructions"]
            assert "invent prices" in prompt
            assert "three health-focused meals" not in prompt
            return Response()

    class Client:
        responses = Responses()

    generator = OpenAIAnswerGenerator(api_key="fake-key", model="test-model", client=Client())
    from app.retrieval.search import SearchResult
    evidence = [SearchResult(
        chunk_id="package-1",
        source_id="adenocarcinoma-package",
        page_type="treatment_package",
        source_url="https://www.placidway.com/package/7903/example",
        title="Adenocarcinoma package",
        text="Outpatient program is $18,995 USD for three weeks.",
        score=1.0,
    )]
    assert "can't verify" in generator.generate("I heard it costs $50. Confirm?", evidence)



def test_price_format_normalizer_repairs_missing_currency_symbols():
    from app.answering.openai_generator import _normalize_price_format

    text = (
        "I can't verify that the package costs `50 from the approved pages. "
        "The page lists prices as `18,995 USD and $30,000 USD."
    )
    normalized = _normalize_price_format(text)
    assert "costs 50" in normalized
    assert "18,995 USD" in normalized
    assert "30,000 USD" in normalized
    assert "`" not in normalized



def test_price_normalizer_handles_unclosed_backticks_and_preserves_usd():
    from app.answering.openai_generator import _normalize_price_format

    text = (
        "I can't verify that the package costs `50 from the approved pages. "
        "The page lists the price as `18,995 USD for outpatient and `30,000 USD for inpatient."
    )
    normalized = _normalize_price_format(text)
    assert "costs 50" in normalized
    assert "18,995 USD" in normalized
    assert "30,000 USD" in normalized
    assert "`" not in normalized


def test_prompt_enforces_strict_price_formatting_and_unverified_claims():
    class Response:
        output_text = json.dumps({"claims": [{"text": "I can't verify that amount from the approved PlacidWay pages.", "chunk_ids": []}]})

    class Responses:
        def create(self, **kwargs):
            prompt = kwargs["instructions"]
            assert "invent prices" in prompt
            assert "outside knowledge" in prompt
            return Response()

    class Client:
        responses = Responses()

    generator = OpenAIAnswerGenerator(api_key="fake-key", model="test-model", client=Client())
    from app.retrieval.search import SearchResult
    evidence = [SearchResult(
        chunk_id="package-1",
        source_id="adenocarcinoma-package",
        page_type="treatment_package",
        source_url="https://www.placidway.com/package/7903/example",
        title="Adenocarcinoma package",
        text="Outpatient program is $18,995 USD for three weeks.",
        score=1.0,
    )]
    answer = generator.generate("I heard it costs $50. Confirm?", evidence)
    assert "can't verify that amount" in answer


def test_prompt_prevents_cross_package_inferences_and_unattributed_medical_claims():
    class Response:
        output_text = json.dumps({"claims": [{"text": "I can't verify that detail from the approved PlacidWay pages.", "chunk_ids": []}]})

    class Responses:
        def create(self, **kwargs):
            prompt = kwargs["instructions"]
            assert "merge details between packages" in prompt
            assert "endorse safety/effectiveness" in prompt
            assert "three health-focused meals" not in prompt
            return Response()

    class Client:
        responses = Responses()

    generator = OpenAIAnswerGenerator(api_key="fake-key", model="test-model", client=Client())
    from app.retrieval.search import SearchResult
    evidence = [SearchResult(
        chunk_id="package-1",
        source_id="adenocarcinoma-package",
        page_type="treatment_package",
        source_url="https://www.placidway.com/package/7903/example",
        title="Adenocarcinoma package",
        text="Outpatient program is three weeks.",
        score=1.0,
    )]
    answer = generator.generate("What is included?", evidence)
    assert "can't verify" in answer



def test_generic_total_cost_question_asks_which_package():
    answer = make_engine().answer(
        "How much money should I expect to pay for the complete treatment program?"
    )
    assert answer.abstained
    assert answer.reason == "clarify_package_for_total_cost"
    assert "Which treatment or package" in answer.text


def test_country_comparison_retrieves_evidence_for_grounded_reasoning():
    class Generator:
        def generate(self, question, evidence, package_key=None):
            assert evidence
            return "The retrieved pages do not establish comparable prices for the same treatment in both countries."

    answer = make_engine(generator=Generator()).answer("Is the treatment cheaper in Mexico or Turkey?")
    assert answer.abstained
    assert answer.reason == "insufficient_comparable_country_pricing"
    assert "Which treatment or package" in answer.text


def test_generic_package_inclusions_retrieves_evidence_before_clarifying():
    class Generator:
        def generate(self, question, evidence, package_key=None):
            assert evidence
            return "The sources cover different packages. Which package do you mean: adenocarcinoma or adrenal cancer?"

    answer = make_engine(generator=Generator()).answer("What is included in the treatment package?")
    assert answer.abstained
    assert answer.reason == "clarify_package_for_inclusions"
    assert "Which treatment package" in answer.text


def test_diabetes_safety_question_is_guarded_before_retrieval():
    class Searcher:
        def search(self, *args, **kwargs):
            raise AssertionError("Personal medical safety questions must not retrieve website chunks")

    answer = AnswerEngine(Searcher()).answer("Is this surgery safe for my diabetes?")
    assert answer.abstained
    assert answer.reason == "medical_safety_not_in_sources"
    assert answer.evidence == ()


def test_unverified_price_claim_retrieves_evidence_without_using_claim_as_fact():
    class Generator:
        def generate(self, question, evidence, package_key=None):
            assert evidence
            assert "$50" in question
            return "I can't verify that amount from the approved PlacidWay pages."

    answer = make_engine(generator=Generator()).answer("I heard it costs $50. Confirm that.")
    assert answer.abstained
    assert answer.reason == "ambiguous_price_target"
    assert "Which PlacidWay treatment or package" in answer.text

def test_named_package_abstains_when_only_generic_price_evidence_exists():
    chunks = [
        {
            "chunk_id": "generic-1",
            "source_id": "alternative-medicine-price-comparison",
            "page_type": "price_comparison",
            "source_url": "https://www.placidway.com/search-medical-pricings/Alternative-Medicine/All/1",
            "title": "Alternative medicine costs",
            "text": "Generic alternative medicine package cost and average price information.",
        }
    ]
    answer = AnswerEngine(KnowledgeSearch(chunks)).answer(
        "How much does the adenocarcinoma treatment package cost?"
    )
    assert answer.abstained
    assert answer.reason == "named_package_evidence_missing"
    assert "different package" in answer.text


def test_country_comparison_country_detection_uses_word_boundaries():
    from app.answering.engine import _countries_named_in_comparison

    assert _countries_named_in_comparison("Is treatment cheaper in Mexico or Turkey?") == ("mexico", "turkey")
    assert _countries_named_in_comparison("Is it cheaper?") == ()


def test_provider_error_reports_unavailable_instead_of_unrelated_extractive_answer():
    class FailingGenerator:
        def generate(self, question, evidence, package_key=None):
            raise RuntimeError("provider failure")

    chunks = [{
        "chunk_id": "anal-table",
        "source_id": "anal-cancer-package",
        "page_type": "treatment_package",
        "source_url": "https://example.com/anal",
        "title": "Anal cancer package",
        "text": (
            "Anal cancer outpatient package pricing.\nProgram Type\nDuration\nTotal Cost (USD)\n"
            "Out-Patient Program\n3 Weeks\n$18,995\nIn-Patient Program\n3 Weeks\n$30,000"
        ),
    }]
    answer = AnswerEngine(KnowledgeSearch(chunks), generator=FailingGenerator()).answer(
        "What is the listed price and duration for the anal cancer outpatient package?"
    )
    assert answer.abstained
    assert "temporarily unavailable" in answer.text
    assert "$18,995" not in answer.text
    assert answer.latency_ms > 0


def test_missing_generator_reports_unavailable_instead_of_extracting_answer():
    answer = make_engine(generator=None).answer("How long is the adenocarcinoma package?")
    assert answer.abstained
    assert answer.reason == "llm_generation_unavailable"
    assert "temporarily unavailable" in answer.text


def test_generation_failure_preserves_privacy_safe_exception_class():
    class ProviderFailure(RuntimeError):
        pass

    class FailingGenerator:
        def generate(self, question, evidence, package_key=None):
            raise ProviderFailure("secret provider details must not be copied")

    answer = make_engine(FailingGenerator()).answer(
        "How long is the adenocarcinoma outpatient program?"
    )

    assert answer.reason == "llm_generation_unavailable"
    assert answer.exception_class == "ProviderFailure"
    assert "secret provider details" not in answer.text



def test_explicit_average_vs_package_price_comparison_keeps_comparison_source_relevant():
    from app.answering.engine import _source_priority
    from app.retrieval.search import SearchResult

    comparison = SearchResult(
        chunk_id="average",
        source_id="alternative-medicine-price-comparison",
        page_type="price_comparison",
        source_url="https://www.placidway.com/search-medical-pricings/Alternative-Medicine/All/1",
        title="Alternative medicine average prices",
        text="Average price listed: 15,032 USD.",
        score=0.4,
    )
    package = SearchResult(
        chunk_id="package",
        source_id="adenocarcinoma-package",
        page_type="treatment_package",
        source_url="https://www.placidway.com/package/7903/example",
        title="Adenocarcinoma package",
        text="Outpatient program: 18,995 USD for three weeks.",
        score=0.5,
    )

    question = "Compare the 15,032 average with the 18,995 outpatient price."
    assert _source_priority(question, comparison) > _source_priority(question, package)


def test_supported_package_aliases_select_the_matching_package_generically():
    from app.answering.engine import _named_package_key

    assert _named_package_key("What are the optional procedures for anal cancer?") == "anal"
    assert _named_package_key("Compare the adrenal package with another option") == "adrenal"
    assert _named_package_key("Tell me about adenocarcinoma treatment") == "adenocarcinoma"
    assert _named_package_key("What is the weather in Cancun?") is None


def test_package_context_ignores_wrong_package_name_in_previous_answer():
    from app.answering.engine import _named_package_key

    resolved = (
        "Follow-up about the same topic as this previous question: "
        "What are the optional procedures for anal cancer?."
        " The previous assistant answer was: The optional procedures for adrenal cancer."
        " The user now asks for price or cost: How much is the second one."
    )
    assert _named_package_key(resolved) == "anal"
