"""Startup readiness checks that expose safe diagnostics only."""
from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import Settings
from app.knowledge.artifact import ArtifactError, validate_artifact


@dataclass(frozen=True)
class HealthReport:
    ready: bool
    retrieval_mode: str = "lexical_bm25"
    source_count: int = 0
    chunk_count: int = 0
    artifact_digest: str | None = None
    error_code: str | None = None
    message: str | None = None


def check_health(settings: Settings) -> HealthReport:
    try:
        metadata = validate_artifact(settings.index_path, settings.manifest_path)
    except (ArtifactError, OSError, ValueError):
        return HealthReport(
            ready=False,
            error_code="artifact_invalid",
            message="The reviewed knowledge artifact is missing or invalid.",
        )
    return HealthReport(
        ready=True,
        source_count=metadata.source_count,
        chunk_count=metadata.chunk_count,
        artifact_digest=metadata.corpus_sha256[:12],
    )
