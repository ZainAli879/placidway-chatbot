"""Run a compliant audit of the seven approved source URLs.

Usage:
    python -m scripts.audit_sources
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config.sources import APPROVED_SOURCES
from app.ingestion.crawler import PoliteSourceCrawler, write_audit_report


def main() -> int:
    crawler = PoliteSourceCrawler(minimum_interval_seconds=1.0)
    results, extracted = crawler.audit_all()
    report_path = write_audit_report(results)

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for source in APPROVED_SOURCES:
        if source.source_id in extracted:
            (raw_dir / f"{source.source_id}.txt").write_text(
                extracted[source.source_id], encoding="utf-8"
            )

    print(f"Audit report: {report_path}")
    print(json.dumps([result.__dict__ for result in results], indent=2))
    print(
        f"Extracted text for {len(extracted)}/{len(APPROVED_SOURCES)} sources. "
        "Review each status before indexing content."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
