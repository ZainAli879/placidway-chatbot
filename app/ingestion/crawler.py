"""Polite, allowlisted crawler and source audit utilities.

This module deliberately does not follow links or crawl outside the seven
approved assessment URLs. It checks robots.txt before fetching source pages,
uses a descriptive User-Agent, and enforces a minimum one-second request gap.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from app.config.sources import (
    ALLOWED_HOST,
    APPROVED_SOURCES,
    ROBOTS_URL,
    USER_AGENT,
    Source,
)


@dataclass
class AuditResult:
    source_id: str
    page_type: str
    url: str
    checked_at_utc: str
    robots_allowed: bool | None
    http_status: int | None
    final_url: str | None
    title: str | None
    extracted_characters: int
    expected_terms_found: list[str]
    status: str
    error: str | None = None


class SourceAuditError(RuntimeError):
    """Raised when a source cannot safely be fetched."""


class PoliteSourceCrawler:
    def __init__(
        self,
        *,
        user_agent: str = USER_AGENT,
        minimum_interval_seconds: float = 1.0,
        timeout_seconds: float = 20.0,
        session: requests.Session | None = None,
    ) -> None:
        if minimum_interval_seconds < 1.0:
            raise ValueError("The assessment requires requests to be about one per second.")
        self.user_agent = user_agent
        self.minimum_interval_seconds = minimum_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml",
            }
        )
        self._last_request_at: float | None = None
        self._robots_parser: RobotFileParser | None = None

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            wait_seconds = self.minimum_interval_seconds - elapsed
            if wait_seconds > 0:
                time.sleep(wait_seconds)
        self._last_request_at = time.monotonic()

    def _get(self, url: str) -> requests.Response:
        self._wait_for_rate_limit()
        response = self.session.get(url, timeout=self.timeout_seconds)
        return response

    def robots_allowed(self, url: str) -> bool:
        """Fetch and evaluate robots.txt; fail closed if policy cannot be checked."""
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
            return False

        if self._robots_parser is None:
            try:
                response = self._get(ROBOTS_URL)
                response.raise_for_status()
                parser = RobotFileParser()
                parser.set_url(ROBOTS_URL)
                parser.parse(response.text.splitlines())
                self._robots_parser = parser
            except requests.RequestException as exc:
                raise SourceAuditError(
                    f"Could not verify robots.txt; refusing to crawl: {exc}"
                ) from exc

        return self._robots_parser.can_fetch(self.user_agent, url)

    @staticmethod
    def extract_page(html: str) -> tuple[str, str]:
        """Return (title, readable text) after removing non-content elements."""
        soup = BeautifulSoup(html, "lxml")

        for element in soup(
            ["script", "style", "noscript", "svg", "iframe", "form", "nav", "footer",
             "button", "select", "option", "input"]
        ):
            element.decompose()

        # Remove common booking/navigation widgets that are sometimes rendered inside
        # the page's main content rather than in a semantic <nav> or <form>.
        widget_pattern = re.compile(
            r"(booking|appointment|timezone|time-zone|calendar|schedule-widget|"
            r"consultation-widget|modal|popup|cookie-banner)",
            re.IGNORECASE,
        )
        for element in list(soup.find_all(True)):
            # Removing a parent also detaches its descendants; BeautifulSoup may
            # set those descendants' attrs to None while they remain in this list.
            if not getattr(element, "attrs", None):
                continue
            attrs = " ".join([
                str(element.get("id", "")),
                " ".join(element.get("class", [])) if isinstance(element.get("class"), list) else str(element.get("class", "")),
                str(element.get("aria-label", "")),
            ])
            if attrs and widget_pattern.search(attrs):
                element.decompose()

        title = ""
        if soup.title:
            title = " ".join(soup.title.get_text(" ", strip=True).split())

        main = soup.find("main") or soup.find("article") or soup.body or soup
        raw_lines = [
            line.strip()
            for line in main.get_text("\n", strip=True).splitlines()
            if line.strip()
        ]
        # Drop residual interface labels and timezone menus even when the site
        # renders them as plain text instead of identifiable HTML widgets.
        ui_line = re.compile(
            r"^(request a video call|select (date|time|time zone)|confirm your virtual consultation|"
            r"book a virtual consultation|visit placidway|skip to main content)$",
            re.IGNORECASE,
        )
        raw_lines = [line for line in raw_lines if not ui_line.match(line)]
        text = "\n".join(raw_lines)
        return title, text

    def audit_source(self, source: Source) -> tuple[AuditResult, str | None]:
        checked_at = datetime.now(timezone.utc).isoformat()
        parsed = urlparse(source.url)

        if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
            return (
                AuditResult(
                    source_id=source.source_id,
                    page_type=source.page_type,
                    url=source.url,
                    checked_at_utc=checked_at,
                    robots_allowed=False,
                    http_status=None,
                    final_url=None,
                    title=None,
                    extracted_characters=0,
                    expected_terms_found=[],
                    status="rejected_not_allowlisted_host",
                    error="URL is outside the approved HTTPS host.",
                ),
                None,
            )

        try:
            if not self.robots_allowed(source.url):
                return (
                    AuditResult(
                        source_id=source.source_id,
                        page_type=source.page_type,
                        url=source.url,
                        checked_at_utc=checked_at,
                        robots_allowed=False,
                        http_status=None,
                        final_url=None,
                        title=None,
                        extracted_characters=0,
                        expected_terms_found=[],
                        status="blocked_by_robots",
                        error="robots.txt does not allow this User-Agent to fetch the URL.",
                    ),
                    None,
                )

            response = self._get(source.url)
            response.raise_for_status()

            final = urlparse(response.url)
            if final.scheme != "https" or final.hostname != ALLOWED_HOST:
                raise SourceAuditError("Redirect left the approved HTTPS host; response rejected.")

            content_type = response.headers.get("Content-Type", "").lower()
            if "html" not in content_type:
                raise SourceAuditError(f"Expected HTML but received Content-Type: {content_type!r}")

            title, text = self.extract_page(response.text)
            found_terms = [
                term for term in source.expected_terms if term.casefold() in text.casefold()
            ]

            if len(text) < 500:
                status = "needs_review_short_extraction"
            elif not found_terms:
                status = "needs_review_expected_terms_missing"
            else:
                status = "fetched_extracted_needs_manual_review"

            result = AuditResult(
                source_id=source.source_id,
                page_type=source.page_type,
                url=source.url,
                checked_at_utc=checked_at,
                robots_allowed=True,
                http_status=response.status_code,
                final_url=response.url,
                title=title or None,
                extracted_characters=len(text),
                expected_terms_found=found_terms,
                status=status,
            )
            return result, text

        except (requests.RequestException, SourceAuditError) as exc:
            return (
                AuditResult(
                    source_id=source.source_id,
                    page_type=source.page_type,
                    url=source.url,
                    checked_at_utc=checked_at,
                    robots_allowed=None,
                    http_status=None,
                    final_url=None,
                    title=None,
                    extracted_characters=0,
                    expected_terms_found=[],
                    status="fetch_or_policy_error",
                    error=str(exc),
                ),
                None,
            )

    def audit_all(self) -> tuple[list[AuditResult], dict[str, str]]:
        results: list[AuditResult] = []
        extracted: dict[str, str] = {}
        for source in APPROVED_SOURCES:
            result, text = self.audit_source(source)
            results.append(result)
            if text is not None:
                extracted[source.source_id] = text
        return results, extracted


def write_audit_report(
    results: list[AuditResult],
    *,
    output_path: str | Path = "data/source_audit.json",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_count": len(results),
        "successful_http_responses": sum(r.http_status is not None for r in results),
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
