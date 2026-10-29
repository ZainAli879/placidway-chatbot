from app.knowledge.indexer import chunk_text, extract_price_mentions
from app.knowledge.schemas import SourceDocument


def test_chunk_text_returns_content_and_respects_limit():
    text = ("First paragraph. " * 50) + "\n\n" + ("Second paragraph. " * 50)
    chunks = chunk_text(text, chunk_size=300, overlap=40)
    assert chunks and all(len(x) <= 300 for x in chunks)
    assert "First paragraph" in " ".join(chunks)
    assert "Second paragraph" in " ".join(chunks)


def test_chunk_text_handles_empty_input():
    assert chunk_text("   \n  ") == []


def test_chunk_text_keeps_heading_with_following_content():
    text = "Pricing\n\nOutpatient program costs $18,995 for three weeks.\n\nInpatient program costs $30,000 for three weeks."
    chunks = chunk_text(text, chunk_size=300, overlap=30)
    assert any("Pricing" in chunk and "Outpatient program costs $18,995" in chunk for chunk in chunks)
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_chunk_text_prefers_sentence_boundaries_for_long_blocks():
    text = ("First complete sentence. " * 20) + ("Second complete sentence. " * 20)
    chunks = chunk_text(text, chunk_size=220, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 220 for chunk in chunks)
    assert all(chunk.endswith(".") or chunk[-1].isspace() for chunk in chunks[:-1])


def test_price_mentions_keep_source_context_and_require_review():
    doc = SourceDocument(
        "test-package", "treatment_package",
        "https://www.placidway.com/package/example", "Example",
        "2026-01-01T00:00:00+00:00",
        "Outpatient program: $18,995 USD for three weeks. Extra services may cost US $500."
    )
    mentions = extract_price_mentions(doc)
    assert len(mentions) >= 2
    assert all(x.source_url == doc.url and x.review_required for x in mentions)
    assert {x.currency for x in mentions} == {"USD"}



def test_chunk_text_does_not_cut_sentences_midway():
    text = "First complete sentence. Second complete sentence. Third complete sentence."
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert chunks == [text]


def test_chunk_text_respects_sentence_boundaries_when_oversized():
    text = ("This is a complete sentence. " * 20) + ("Another complete sentence! " * 20)
    chunks = chunk_text(text, chunk_size=220, overlap=20)
    assert all(len(chunk) <= 220 for chunk in chunks)
    assert all(chunk.endswith((".", "!")) for chunk in chunks[:-1])


def test_clean_source_text_removes_booking_interface_noise():
    from app.knowledge.indexer import clean_source_text
    text = "Useful clinic details.\nRequest a video call\nSelect time zone\nPacific Time (US and Canada)\nPackage price is 18,995 USD."
    cleaned = clean_source_text(text)
    assert "Useful clinic details." in cleaned
    assert "Package price is 18,995 USD." in cleaned
    assert "Request a video call" not in cleaned
    assert "Pacific Time" not in cleaned
