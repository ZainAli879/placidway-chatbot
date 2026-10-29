import json
import pytest

from app.answering.openai_generator import (
    EmptyModelResponseError,
    GroundingValidationError,
    InvalidStructuredOutputError,
    IncompleteStructuredOutputError,
    OpenAIAnswerGenerator,
    _filter_cost_evidence,
    _output_token_budget,
)
from app.retrieval.search import SearchResult


def result(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        source_id="adenocarcinoma-package",
        page_type="treatment_package",
        source_url="https://example.com/package",
        title="Alternative Adenocarcinoma Cancer Treatment",
        text=text,
        score=0.5,
    )


def generator_returning(
    output_text: str, *, grounding_validation_enabled: bool = True
) -> OpenAIAnswerGenerator:
    class Response:
        pass

    Response.output_text = output_text

    class Responses:
        def create(self, **kwargs):
            return Response()

    class Client:
        responses = Responses()

    return OpenAIAnswerGenerator(
        api_key="fake",
        model="test-model",
        client=Client(),
        grounding_validation_enabled=grounding_validation_enabled,
    )


def test_general_cost_followup_keeps_base_prices_and_filters_addon_price_chunk():
    base = result(
        "base",
        "Cost of program. Total Cost (USD): Out-Patient $18,995; In-Patient $30,000.",
    )
    addons = result(
        "addons",
        "Lymphocytes Activated Killer Cells (LAK): $2,600. Excluded Services and Contingency Costs.",
    )

    filtered = _filter_cost_evidence("Follow-up about same topic. What about the cost?", [base, addons])

    assert [item.chunk_id for item in filtered] == ["base"]


def test_explicit_addon_cost_question_keeps_addon_evidence():
    base = result("base", "Total Cost (USD): Out-Patient $18,995; In-Patient $30,000.")
    addons = result(
        "addons",
        "Optional specialized procedures. Lymphocytes Activated Killer Cells (LAK): $2,600.",
    )

    filtered = _filter_cost_evidence(
        "What are the optional add-on procedure costs?",
        [base, addons],
    )

    assert [item.chunk_id for item in filtered] == ["base", "addons"]


def test_non_cost_question_keeps_all_evidence():
    base = result("base", "Included in the outpatient program.")
    addons = result("addons", "Optional specialized procedures and extra charges.")

    filtered = _filter_cost_evidence("What is included in the package?", [base, addons])

    assert [item.chunk_id for item in filtered] == ["base", "addons"]


def test_prompt_handles_price_comparisons_and_page_types_without_inventing_content():
    class Response:
        output_text = json.dumps({"claims": [{"text": "The listed price is 18,995 USD.", "chunk_ids": ["base"]}]})

    class Responses:
        def create(self, **kwargs):
            instructions = kwargs["instructions"]
            assert "every factual claim" in instructions
            assert "chunk_ids" in instructions
            assert "all materially relevant requested facts" in instructions
            assert "Do not omit a relevant requested fact" in instructions
            assert "Answer only the details the visitor requested" in instructions
            assert "combine related list items into one claim" in instructions
            assert "three health-focused meals" not in instructions
            assert kwargs["max_output_tokens"] == 250
            assert kwargs["text"]["format"]["type"] == "json_schema"
            assert kwargs["text"]["format"]["strict"] is True
            assert "reasoning" not in kwargs
            return Response()

    class Client:
        responses = Responses()

    generator = OpenAIAnswerGenerator(api_key="fake-key", model="test-model", client=Client())
    answer = generator.generate("Compare these two listed prices.", [result("base", "Price: 18,995 USD.")])
    assert "18,995 USD" in answer
    assert "https://example.com/package" in answer


def test_generator_caps_chunks_and_evidence_characters():
    captured = {}

    class Response:
        output_text = json.dumps({"claims": [{"text": "The page lists this information.", "chunk_ids": ["c0"]}]})

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return Response()

    class Client:
        responses = Responses()

    evidence = [result(f"c{i}", "information " * 1000) for i in range(5)]
    generator = OpenAIAnswerGenerator(api_key="fake-key", client=Client())
    generator.generate("What information is listed?", evidence)

    supplied = captured["input"].split("Approved-page evidence:\n", 1)[1]
    assert "Chunk ID: c2" in supplied
    assert "Chunk ID: c4" in supplied
    evidence_chars = sum(
        len(block.split("Untrusted page text:\n", 1)[1])
        for block in supplied.split("\n\nChunk ID:")
    )
    assert evidence_chars <= 6000


def test_default_model_is_4_1_mini_and_explicit_override_is_preserved(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    client = type("Client", (), {"responses": object()})()
    assert OpenAIAnswerGenerator(api_key="fake", client=client).model == "gpt-4.1-mini"
    assert OpenAIAnswerGenerator(api_key="fake", model="custom-model", client=client).model == "custom-model"


def test_output_budget_is_not_reduced_for_focused_or_comparison_questions():
    assert _output_token_budget("What is the listed outpatient price and duration?") == 250
    assert _output_token_budget("Compare the outpatient and inpatient packages") == 250
    assert _output_token_budget("List all optional procedures and their costs") == 250
    assert _output_token_budget("What is included in the outpatient package?") == 250
    assert _output_token_budget("List the procedures available for anal cancer") == 250
    assert _output_token_budget("Is the price of optional procedures included?") == 250


def test_empty_model_output_has_distinct_failure_type():
    with pytest.raises(EmptyModelResponseError):
        generator_returning("").generate("What is listed?", [result("c0", "Listed information.")])


def test_invalid_json_output_has_distinct_failure_type():
    with pytest.raises(InvalidStructuredOutputError):
        generator_returning("not-json").generate(
            "What is listed?", [result("c0", "Listed information.")]
        )


def test_grounding_rejection_has_distinct_failure_type():
    output = json.dumps({
        "claims": [{"text": "The unsupported price is 99,999 USD.", "chunk_ids": ["c0"]}]
    })
    with pytest.raises(GroundingValidationError):
        generator_returning(output, grounding_validation_enabled=True).generate(
            "What is listed?", [result("c0", "Listed information.")]
        )


def test_grounding_validation_can_be_explicitly_disabled_but_sources_still_render():
    output = json.dumps({
        "claims": [{"text": "The unsupported price is 99,999 USD.", "chunk_ids": ["c0"]}]
    })

    answer = generator_returning(output, grounding_validation_enabled=False).generate(
        "What is listed?", [result("c0", "Listed information.")]
    )

    assert "99,999 USD" in answer
    assert "https://example.com/package" in answer


def test_grounding_validation_is_enabled_by_default():
    output = json.dumps({
        "claims": [{"text": "The unsupported price is 99,999 USD.", "chunk_ids": ["c0"]}]
    })
    with pytest.raises(GroundingValidationError):
        generator_returning(output).generate(
            "What is listed?", [result("c0", "Listed information.")]
        )


def test_generator_uses_configured_evidence_budgets():
    captured = {}

    class Response:
        output_text = json.dumps({
            "claims": [{"text": "The page lists information.", "chunk_ids": ["c0"]}]
        })

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return Response()

    client = type("Client", (), {"responses": Responses()})()
    evidence = [result(f"c{i}", "information " * 200) for i in range(4)]
    generator = OpenAIAnswerGenerator(
        api_key="fake",
        client=client,
        max_evidence_chunks=2,
        max_evidence_chars=700,
        grounding_validation_enabled=False,
    )
    generator.generate("What information is listed?", evidence)

    supplied = captured["input"].split("Approved-page evidence:\n", 1)[1]
    assert "Chunk ID: c1" in supplied
    assert "Chunk ID: c2" not in supplied
    assert len(supplied) < 1400


def test_incomplete_structured_output_retries_once_with_larger_budget():
    calls = []

    class Response:
        def __init__(self, status, output_text):
            self.status = status
            self.output_text = output_text
            self.incomplete_details = type("Details", (), {"reason": "max_output_tokens"})()

    responses = iter([
        Response("incomplete", '{"claims":['),
        Response("completed", json.dumps({
            "claims": [{"text": "The page lists information.", "chunk_ids": ["c0"]}]
        })),
    ])

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return next(responses)

    client = type("Client", (), {"responses": Responses()})()
    answer = OpenAIAnswerGenerator(api_key="fake", client=client).generate(
        "What is listed?", [result("c0", "The page lists information.")]
    )

    assert len(calls) == 2
    assert calls[0]["max_output_tokens"] == 250
    assert calls[1]["max_output_tokens"] == 400
    assert "The page lists information" in answer


def test_second_incomplete_structured_output_has_specific_error():
    class Response:
        status = "incomplete"
        output_text = '{"claims":['
        incomplete_details = type("Details", (), {"reason": "max_output_tokens"})()

    class Responses:
        def create(self, **kwargs):
            return Response()

    client = type("Client", (), {"responses": Responses()})()
    with pytest.raises(IncompleteStructuredOutputError):
        OpenAIAnswerGenerator(api_key="fake", client=client).generate(
            "What is listed?", [result("c0", "The page lists information.")]
        )
