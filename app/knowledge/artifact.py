"""Build, approve, and validate immutable knowledge artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Sequence

from app.config.sources import APPROVED_SOURCES

SCHEMA_VERSION = 1


class ArtifactError(ValueError):
    """The deployed knowledge artifact is incomplete, altered, or unreviewed."""


@dataclass(frozen=True)
class ArtifactMetadata:
    schema_version: int
    corpus_sha256: str
    source_count: int
    chunk_count: int
    reviewed_by: str
    reviewed_at_utc: str


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Knowledge corpus is not valid JSONL: {exc.__class__.__name__}") from exc


def _validate_rows(rows: Sequence[dict]) -> tuple[str, ...]:
    expected = {source.source_id: source.url for source in APPROVED_SOURCES}
    found: set[str] = set()
    for index, row in enumerate(rows, start=1):
        source_id = row.get("source_id")
        if source_id not in expected:
            raise ArtifactError(f"Chunk {index} has an unknown source ID")
        if row.get("source_url") != expected[source_id]:
            raise ArtifactError(f"Chunk {index} has an unapproved source URL")
        required = ("chunk_id", "page_type", "title", "text")
        if any(not row.get(field) for field in required):
            raise ArtifactError(f"Chunk {index} is missing required fields")
        found.add(source_id)
    missing = sorted(set(expected) - found)
    if missing:
        raise ArtifactError(f"Knowledge corpus is missing approved sources: {', '.join(missing)}")
    return tuple(sorted(found))


def build_manifest(corpus_path: Path, chunks: Sequence[dict], *, built_at_utc: str | None = None) -> dict:
    sources = _validate_rows(chunks)
    return {
        "schema_version": SCHEMA_VERSION,
        "built_at_utc": built_at_utc or datetime.now().astimezone().isoformat(),
        "corpus_sha256": _digest(corpus_path),
        "source_ids": list(sources),
        "source_count": len(sources),
        "chunk_count": len(chunks),
        "review_status": "pending",
    }


def approve_artifact(
    corpus_path: Path,
    manifest_path: Path,
    reviewer: str,
    now: datetime,
) -> dict:
    reviewer = reviewer.strip()
    if not reviewer:
        raise ArtifactError("A non-empty reviewer identity is required")
    rows = _read_rows(corpus_path)
    manifest = build_manifest(corpus_path, rows)
    manifest.update(
        review_status="approved",
        reviewed_by=reviewer,
        reviewed_at_utc=now.isoformat(),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_artifact(corpus_path: Path, manifest_path: Path) -> ArtifactMetadata:
    rows = _read_rows(corpus_path)
    sources = _validate_rows(rows)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Knowledge manifest is not valid JSON: {exc.__class__.__name__}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError("Knowledge manifest schema version is unsupported")
    if manifest.get("corpus_sha256") != _digest(corpus_path):
        raise ArtifactError("Knowledge corpus digest does not match its reviewed manifest")
    if manifest.get("source_ids") != list(sources) or manifest.get("source_count") != len(sources):
        raise ArtifactError("Knowledge manifest source metadata does not match the corpus")
    if manifest.get("chunk_count") != len(rows):
        raise ArtifactError("Knowledge manifest chunk count does not match the corpus")
    if manifest.get("review_status") != "approved" or not manifest.get("reviewed_by"):
        raise ArtifactError("Knowledge artifact has not been approved")
    return ArtifactMetadata(
        schema_version=SCHEMA_VERSION,
        corpus_sha256=manifest["corpus_sha256"],
        source_count=len(sources),
        chunk_count=len(rows),
        reviewed_by=manifest["reviewed_by"],
        reviewed_at_utc=manifest["reviewed_at_utc"],
    )
