from app.answering.grounding import validate_generation
from app.retrieval.search import SearchResult


def evidence(chunk_id="pkg-1", source_id="adenocarcinoma-package", text="The package page lists an outpatient price of 18,995 USD."):
    return SearchResult(
        chunk_id=chunk_id,
        source_id=source_id,
        page_type="treatment_package",
        source_url="https://www.placidway.com/package/example",
        title="Package",
        text=text,
        score=1.0,
    )


def test_valid_generation_renders_claim_level_source_link():
    rendered = validate_generation(
        {"claims": [{"text": "The outpatient price listed is 18,995 USD.", "chunk_ids": ["pkg-1"]}]},
        [evidence()],
        "adenocarcinoma",
    )

    assert rendered is not None
    assert "18,995 USD" in rendered
    assert "[Package](https://www.placidway.com/package/example)" in rendered


def test_factual_claim_without_citation_is_rejected():
    assert validate_generation(
        {"claims": [{"text": "The outpatient price is 18,995 USD.", "chunk_ids": []}]},
        [evidence()],
        "adenocarcinoma",
    ) is None


def test_unknown_or_cross_package_citation_is_rejected():
    assert validate_generation(
        {"claims": [{"text": "The outpatient price is 18,995 USD.", "chunk_ids": ["unknown"]}]},
        [evidence()],
        "adenocarcinoma",
    ) is None
    assert validate_generation(
        {"claims": [{"text": "The adrenal package has this information.", "chunk_ids": ["other"]}]},
        [evidence("other", "adrenal-cancer-package", "The adrenal package has this information.")],
        "adenocarcinoma",
    ) is None


def test_unsupported_price_is_rejected_even_with_valid_chunk_id():
    assert validate_generation(
        {"claims": [{"text": "The outpatient price is 99,999 USD.", "chunk_ids": ["pkg-1"]}]},
        [evidence()],
        "adenocarcinoma",
    ) is None


def test_limitation_sentence_may_be_uncited():
    assert validate_generation(
        {"claims": [{"text": "I could not verify that detail from the approved pages.", "chunk_ids": []}]},
        [evidence()],
        None,
    ) == "I could not verify that detail from the approved pages."


def test_rejects_changed_duration_and_appended_medical_claim():
    duration = evidence(text="The outpatient program lasts three weeks.")
    assert validate_generation(
        {"claims": [{"text": "The outpatient program lasts thirty weeks.", "chunk_ids": ["pkg-1"]}]},
        [duration], "adenocarcinoma",
    ) is None


def test_accepts_equivalent_number_word_and_digit():
    duration = evidence(text="The follow-up consultation is 3 months after discharge.")
    rendered = validate_generation(
        {"claims": [{"text": "The follow-up consultation is three months after discharge.", "chunk_ids": ["pkg-1"]}]},
        [duration], "adenocarcinoma",
    )
    assert rendered is not None
    assert validate_generation(
        {"claims": [{"text": "The outpatient program lasts three weeks. It cures cancer.", "chunk_ids": ["pkg-1"]}]},
        [duration], "adenocarcinoma",
    ) is None


def test_accepts_plural_variants_of_terms_present_in_cited_evidence():
    videos = evidence(
        chunk_id="videos-1",
        source_id="itc-clinic-videos",
        text=(
            "The video page covers an alternative cancer treatment journey. Success story. "
            "Detailed overview of cancer treatment procedures and a protocol guide."
        ),
    )
    rendered = validate_generation(
        {
            "claims": [{
                "text": (
                    "The videos include treatment journeys, success stories, detailed "
                    "overviews of cancer treatments and procedure guides."
                ),
                "chunk_ids": ["videos-1"],
            }]
        },
        [videos],
        None,
    )

    assert rendered is not None


def test_accepts_supported_comma_prices_and_month_name_for_iso_date():
    comparison = evidence(
        chunk_id="comparison-1",
        source_id="alternative-medicine-price-comparison",
        text=(
            "Alternative Medicine Cost Abroad. Alternative Medicine costs on average "
            "$15032, with most clinics quoting between $64 and $30000. Based on verified "
            "quotes from 4+ clinics, last updated 2026-06-03. Price Range (USD): $64 - $30000."
        ),
    )
    rendered = validate_generation(
        {
            "claims": [{
                "text": (
                    "The alternative medicine price range abroad is between $64 and $30,000, "
                    "with an average cost of $15,032, based on verified quotes from multiple "
                    "clinics as of June 2026."
                ),
                "chunk_ids": ["comparison-1"],
            }]
        },
        [comparison],
        None,
    )

    assert rendered is not None


def test_accepts_grounded_price_paraphrase_without_rigid_word_overlap():
    comparison = evidence(
        chunk_id="comparison-1",
        source_id="alternative-medicine-price-comparison",
        text="Price Range (USD): $64 - $30000.",
    )
    rendered = validate_generation(
        {
            "claims": [{
                "text": "The lowest listed charge is $64 and the highest is $30,000.",
                "chunk_ids": ["comparison-1"],
            }]
        },
        [comparison],
        None,
    )

    assert rendered is not None


def test_limitation_prefix_cannot_hide_an_uncited_factual_claim():
    assert validate_generation(
        {"claims": [{"text": "Please confirm the price. The treatment cures cancer.", "chunk_ids": []}]},
        [evidence()], None,
    ) is None


def test_modal_may_is_not_interpreted_as_calendar_month():
    rendered = validate_generation(
        {"claims": [{
            "text": "The page says procedures may be optional.",
            "chunk_ids": ["pkg-1"],
        }]},
        [evidence(text="The page lists optional procedures.")],
        "adenocarcinoma",
    )
    assert rendered is not None


def test_equivalent_whole_dollar_prices_are_accepted():
    rendered = validate_generation(
        {"claims": [{
            "text": "The listed price is $18,995.00.",
            "chunk_ids": ["pkg-1"],
        }]},
        [evidence(text="The listed price is $18,995.")],
        "adenocarcinoma",
    )
    assert rendered is not None


def test_compound_number_words_cannot_invent_a_duration():
    assert validate_generation(
        {"claims": [{
            "text": "The program lasts twenty-three weeks.",
            "chunk_ids": ["pkg-1"],
        }]},
        [evidence(text="The program lasts three weeks and includes twenty consultations.")],
        "adenocarcinoma",
    ) is None


def test_medical_caveat_is_allowed_but_unsupported_safety_claim_is_rejected():
    caveat = validate_generation(
        {"claims": [{
            "text": "The page does not guarantee results.",
            "chunk_ids": ["pkg-1"],
        }]},
        [evidence(text="The page does not guarantee results.")],
        "adenocarcinoma",
    )
    assert caveat is not None
    assert validate_generation(
        {"claims": [{
            "text": "The treatment is completely safe and will remove your tumor.",
            "chunk_ids": ["pkg-1"],
        }]},
        [evidence(text="A consultation with the clinic is available.")],
        "adenocarcinoma",
    ) is None
