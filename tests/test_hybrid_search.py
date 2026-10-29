from app.retrieval.hybrid_search import HybridKnowledgeSearch


def chunks():
    return [
        {
            "chunk_id": "lexical",
            "source_id": "source-a",
            "page_type": "profile",
            "source_url": "https://example.com/a",
            "title": "Exact adrenal terminology",
            "text": "The adrenal outpatient program is described here.",
        },
        {
            "chunk_id": "semantic",
            "source_id": "source-b",
            "page_type": "profile",
            "source_url": "https://example.com/b",
            "title": "Related care",
            "text": "A conceptually related passage without the query vocabulary.",
        },
    ]


class FakeCollection:
    def query(self, **kwargs):
        assert kwargs["query_texts"] == ["adrenal outpatient"]
        return {
            "ids": [["semantic", "lexical"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{}, {}]],
        }


def test_hybrid_search_fuses_and_marks_lexical_and_vector_provenance():
    search = HybridKnowledgeSearch(chunks(), collection=FakeCollection())

    results = search.search("adrenal outpatient", top_k=2)

    assert {item.chunk_id for item in results} == {"lexical", "semantic"}
    by_id = {item.chunk_id: item for item in results}
    assert by_id["lexical"].retrieval_sources == ("lexical", "vector")
    assert by_id["semantic"].retrieval_sources == ("vector",)
    assert results[0].chunk_id == "lexical"


def test_hybrid_search_deduplicates_results_found_by_both_legs():
    search = HybridKnowledgeSearch(chunks(), collection=FakeCollection())

    results = search.search("adrenal outpatient", top_k=10)

    assert [item.chunk_id for item in results].count("lexical") == 1


def test_hybrid_search_handles_empty_query_without_calling_collection():
    class ExplodingCollection:
        def query(self, **kwargs):
            raise AssertionError("empty queries must not reach Chroma")

    search = HybridKnowledgeSearch(chunks(), collection=ExplodingCollection())

    assert search.search("   ", top_k=5) == []
    assert search.search("adrenal", top_k=0) == []
