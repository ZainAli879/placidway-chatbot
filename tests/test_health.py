import json
from datetime import datetime, timezone

from app.config.settings import Settings
from app.knowledge.artifact import approve_artifact, build_manifest
from app.config.sources import APPROVED_SOURCES
from app.runtime.health import check_health


def test_health_reports_ready_for_valid_reviewed_artifact(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    manifest = tmp_path / "manifest.json"
    rows = [
        {
            "chunk_id": f"{source.source_id}-1", "source_id": source.source_id,
            "page_type": source.page_type, "source_url": source.url,
            "title": source.source_id, "text": "Reviewed source information.",
        }
        for source in APPROVED_SOURCES
    ]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest.write_text(json.dumps(build_manifest(corpus, rows)), encoding="utf-8")
    approve_artifact(corpus, manifest, "reviewer", datetime.now(timezone.utc))
    settings = Settings.from_sources({
        "KNOWLEDGE_INDEX_PATH": str(corpus),
        "KNOWLEDGE_MANIFEST_PATH": str(manifest),
    })

    report = check_health(settings)

    assert report.ready
    assert report.source_count == 7
    assert report.retrieval_mode == "lexical_bm25"


def test_health_reports_safe_error_for_unreviewed_artifact(tmp_path):
    settings = Settings.from_sources({
        "KNOWLEDGE_INDEX_PATH": str(tmp_path / "missing.jsonl"),
        "KNOWLEDGE_MANIFEST_PATH": str(tmp_path / "missing-manifest.json"),
    })
    report = check_health(settings)
    assert not report.ready
    assert report.error_code == "artifact_invalid"
    assert str(tmp_path) not in (report.message or "")
