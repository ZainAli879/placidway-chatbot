from app.retrieval.search import KnowledgeSearch

def test_search_returns_relevant_source_evidence():
    rows = [
        {"chunk_id":"package-1","source_id":"adenocarcinoma-package","page_type":"treatment_package","source_url":"https://www.placidway.com/package/7903/example","title":"Adenocarcinoma package","text":"The outpatient adenocarcinoma program lasts three weeks."},
        {"chunk_id":"comparison-1","source_id":"alternative-medicine-price-comparison","page_type":"price_comparison","source_url":"https://www.placidway.com/search-medical-pricings/Alternative-Medicine/All/1","title":"Alternative medicine comparison","text":"Compare general alternative medicine prices in different countries."}
    ]
    results = KnowledgeSearch(rows).search("adenocarcinoma outpatient program", top_k=2)
    assert results and results[0].source_id == "adenocarcinoma-package"
    assert results[0].source_url.endswith("/7903/example")

def test_search_returns_empty_for_unmatched_query():
    rows = [{"chunk_id":"a","source_id":"clinic","page_type":"clinic_profile","source_url":"https://www.placidway.com/profile/example","title":"Clinic","text":"Clinic information in Tijuana."}]
    assert KnowledgeSearch(rows).search("weather in Iceland", top_k=3) == []

def test_search_ties_are_ordered_by_chunk_id():
    rows = [
        {"chunk_id":"z","source_id":"s","page_type":"p","source_url":"https://example.com","title":"Z","text":"same treatment detail"},
        {"chunk_id":"a","source_id":"s","page_type":"p","source_url":"https://example.com","title":"A","text":"same treatment detail"},
    ]
    results = KnowledgeSearch(rows).search("treatment detail", top_k=2)
    assert [item.chunk_id for item in results] == ["a", "z"]
