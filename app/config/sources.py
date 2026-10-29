"""Explicit allowlist of source pages approved for the PlacidWay assessment.

Do not add discovered links automatically. The assessment limits crawling to these
seven URLs. Review the source list before making changes.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    source_id: str
    page_type: str
    url: str
    expected_terms: tuple[str, ...]


APPROVED_SOURCES: tuple[Source, ...] = (
    Source(
        source_id="itc-clinic-profile",
        page_type="clinic_profile",
        url="https://www.placidway.com/profile/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
        expected_terms=("Immunity Therapy Center", "Tijuana"),
    ),
    Source(
        source_id="adenocarcinoma-package",
        page_type="treatment_package",
        url="https://www.placidway.com/package/7903/Alternative-Adenocarcinoma-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
        expected_terms=("adenocarcinoma", "Tijuana"),
    ),
    Source(
        source_id="adrenal-cancer-package",
        page_type="treatment_package",
        url="https://www.placidway.com/package/7902/Alternative-Adrenal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
        expected_terms=("adrenal", "Tijuana"),
    ),
    Source(
        source_id="anal-cancer-package",
        page_type="treatment_package",
        url="https://www.placidway.com/package/7901/Alternative-Anal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
        expected_terms=("anal cancer", "Tijuana"),
    ),
    Source(
        source_id="itc-clinic-videos",
        page_type="clinic_videos",
        url="https://www.placidway.com/profile-videos/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
        expected_terms=("ITC", "video"),
    ),
    Source(
        source_id="itc-price-list",
        page_type="price_list",
        url="https://www.placidway.com/price-list/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
        expected_terms=("price", "USD"),
    ),
    Source(
        source_id="alternative-medicine-price-comparison",
        page_type="price_comparison",
        url="https://www.placidway.com/search-medical-pricings/Alternative-Medicine/All/1",
        expected_terms=("alternative medicine", "price"),
    ),
)

ALLOWED_HOST = "www.placidway.com"
ROBOTS_URL = "https://www.placidway.com/robots.txt"
USER_AGENT = (
    "PlacidWayAssessmentBot/1.0 "
    "(source-limited research chatbot; respectful crawling)"
)
