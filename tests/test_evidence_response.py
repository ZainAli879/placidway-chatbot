from app.answering.evidence_response import (
    build_evidence_response, is_list_style_question, select_evidence,
)
from app.retrieval.search import SearchResult


def row(chunk_id, source_id, text, score=1.0):
    return SearchResult(chunk_id, source_id, "treatment_package", f"https://www.placidway.com/{source_id}", source_id, text, score)


def test_select_evidence_isolates_named_package_and_caps_size():
    candidates = [
        row("a1", "adenocarcinoma-package", "adenocarcinoma duration three weeks " * 100, 3),
        row("x1", "adrenal-cancer-package", "adenocarcinoma unrelated cross package", 4),
        row("a2", "adenocarcinoma-package", "adenocarcinoma outpatient details", 2),
    ]
    selected = select_evidence("adenocarcinoma duration", candidates, "adenocarcinoma")
    assert [item.chunk_id for item in selected] == ["a1"]
    assert sum(len(item.text) for item in selected) <= 6000


def test_list_question_can_expand_to_five_relevant_chunks():
    candidates = [
        row(f"a{i}", "adenocarcinoma-package", f"Optional procedure {i} detail and cost.", 10 - i)
        for i in range(6)
    ]
    selected = select_evidence(
        "List all optional procedures and their costs for adenocarcinoma",
        candidates,
        "adenocarcinoma",
        max_chunks=5,
    )
    assert len(selected) == 5


def test_multipart_question_keeps_separate_price_and_duration_chunks():
    candidates = [
        SearchResult("price", "adenocarcinoma-package", "treatment_package", "https://example.com", "Adenocarcinoma price and duration", "The outpatient price is 18,995 USD.", 2),
        SearchResult("duration", "adenocarcinoma-package", "treatment_package", "https://example.com", "Adenocarcinoma price and duration", "The outpatient duration is three weeks.", 1),
    ]
    selected = select_evidence(
        "What are the outpatient price and duration for adenocarcinoma?",
        candidates, "adenocarcinoma", max_chunks=5,
    )
    assert [item.chunk_id for item in selected] == ["price", "duration"]


def test_list_classifier_distinguishes_enumeration_from_yes_no_question():
    assert is_list_style_question("List the procedures available for anal cancer")
    assert is_list_style_question("What is included in the outpatient package?")
    assert not is_list_style_question("Is the price of optional procedures included?")


def test_evidence_response_uses_only_matching_safe_sentences_and_links():
    evidence = [row("a1", "adenocarcinoma-package", "The outpatient program lasts three weeks. Promotional cure guaranteed for everyone.")]
    response = build_evidence_response("How long is the outpatient program?", evidence)
    assert response is not None
    assert "three weeks" in response
    assert "cure guaranteed" not in response
    assert "https://www.placidway.com/adenocarcinoma-package" in response


def test_evidence_response_abstains_when_no_sentence_matches():
    evidence = [row("a1", "adenocarcinoma-package", "The clinic is located in Tijuana.")]
    assert build_evidence_response("What is the outpatient duration?", evidence) is None


def test_evidence_selection_never_slices_a_price_or_sentence():
    evidence = [row("a1", "adenocarcinoma-package", "The price is $18,995 USD.")]
    selected = select_evidence("price", evidence, "adenocarcinoma", max_chars=16)
    assert selected == ()


def test_evidence_response_requires_the_requested_attribute():
    evidence = [row("a1", "adenocarcinoma-package", "The outpatient program is in Tijuana.")]
    assert build_evidence_response("What is the outpatient success rate?", evidence) is None


def test_evidence_response_extracts_price_and_duration_from_package_table():
    table = row(
        "anal-1", "anal-cancer-package",
        "Program Type\nDuration\nTotal Cost (USD)\nOut-Patient Program\n3 Weeks\n$18,995\n"
        "In-Patient Program\n3 Weeks\n$30,000",
    )
    evidence = [table, row("anal-2", "anal-cancer-package", table.text)]
    response = build_evidence_response(
        "What is the listed price and duration for the anal cancer outpatient package?",
        evidence,
    )
    assert response is not None
    assert "Out-Patient Program" in response
    assert "3 Weeks" in response
    assert "$18,995" in response
    assert response.count("$18,995") == 1
