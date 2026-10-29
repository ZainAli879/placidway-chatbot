from app.config.sources import ALLOWED_HOST, APPROVED_SOURCES


def test_exactly_seven_approved_sources():
    assert len(APPROVED_SOURCES) == 7


def test_source_urls_are_unique():
    urls = [source.url for source in APPROVED_SOURCES]
    assert len(urls) == len(set(urls))


def test_all_sources_use_approved_https_host():
    from urllib.parse import urlparse

    for source in APPROVED_SOURCES:
        parsed = urlparse(source.url)
        assert parsed.scheme == "https"
        assert parsed.hostname == ALLOWED_HOST


def test_source_ids_are_unique():
    ids = [source.source_id for source in APPROVED_SOURCES]
    assert len(ids) == len(set(ids))
