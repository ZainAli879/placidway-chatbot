import json
from datetime import datetime, timezone

import pytest

from app.config.sources import APPROVED_SOURCES
from app.knowledge.artifact import (
    ArtifactError,
    approve_artifact,
    build_manifest,
    validate_artifact,
)


def _rows():
    return [
        {
            "chunk_id": f"{source.source_id}-1",
            "source_id": source.source_id,
            "page_type": source.page_type,
            "source_url": source.url,
            "title": source.source_id,
            "chunk_index": 0,
            "text": f"Evidence from {source.source_id}.",
        }
        for source in APPROVED_SOURCES
    ]


def _write_corpus(path, rows=None):
    rows = rows or _rows()
    path.write_bytes(b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows))
    return rows


def test_artifact_requires_all_approved_sources(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    rows = _write_corpus(corpus, _rows()[:-1])

    with pytest.raises(ArtifactError, match="missing approved sources"):
        build_manifest(corpus, rows)


def test_artifact_rejects_unknown_source_url(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    rows = _rows()
    rows[0]["source_url"] = "https://evil.example/source"
    _write_corpus(corpus, rows)

    with pytest.raises(ArtifactError, match="URL"):
        build_manifest(corpus, rows)


def test_approval_requires_reviewer_and_exact_digest(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.json"
    rows = _write_corpus(corpus)
    manifest_path.write_text(json.dumps(build_manifest(corpus, rows)), encoding="utf-8")

    with pytest.raises(ArtifactError, match="reviewer"):
        approve_artifact(corpus, manifest_path, "", datetime.now(timezone.utc))

    approved = approve_artifact(
        corpus,
        manifest_path,
        "quality-team@example.com",
        datetime(2026, 10, 28, tzinfo=timezone.utc),
    )
    metadata = validate_artifact(corpus, manifest_path)

    assert approved["reviewed_by"] == "quality-team@example.com"
    assert metadata.reviewed_by == "quality-team@example.com"
    assert metadata.source_count == 7
    assert metadata.chunk_count == 7

    corpus.write_bytes(corpus.read_bytes() + b"\n")
    with pytest.raises(ArtifactError, match="digest"):
        validate_artifact(corpus, manifest_path)


def test_rebuild_manifest_is_always_unreviewed(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    rows = _write_corpus(corpus)

    manifest = build_manifest(corpus, rows)

    assert "reviewed_by" not in manifest
    assert manifest["review_status"] == "pending"


def test_artifact_rejects_malformed_jsonl(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    manifest = tmp_path / "manifest.json"
    corpus.write_text("not-json\n", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactError, match="JSON"):
        validate_artifact(corpus, manifest)
