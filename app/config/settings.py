"""Validated runtime configuration with secret-safe errors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class SettingsError(ValueError):
    """A public-safe configuration error that never includes secret values."""


def _value(
    name: str,
    environ: Mapping[str, str],
    secrets: Mapping[str, object] | None,
    default: str,
) -> str:
    if name in environ:
        return str(environ[name]).strip()
    if secrets is not None and name in secrets:
        return str(secrets[name]).strip()
    return default


def _integer(
    name: str,
    environ: Mapping[str, str],
    secrets: Mapping[str, object] | None,
    default: int,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    raw = _value(name, environ, secrets, str(default))
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer") from exc
    if parsed < minimum or (maximum is not None and parsed > maximum):
        bounds = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        raise SettingsError(f"{name} must be {bounds}")
    return parsed


def _boolean(
    name: str,
    environ: Mapping[str, str],
    secrets: Mapping[str, object] | None,
    default: bool,
) -> bool:
    raw = _value(name, environ, secrets, str(default)).casefold()
    if raw in {"true", "1", "yes", "on"}:
        return True
    if raw in {"false", "0", "no", "off"}:
        return False
    raise SettingsError(f"{name} must be true or false")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    index_path: Path
    manifest_path: Path
    chroma_persist_directory: Path
    grounding_validation_enabled: bool
    max_question_chars: int
    max_evidence_chunks: int
    max_evidence_chars: int
    session_requests: int
    session_window_seconds: int
    daily_generation_limit: int

    @property
    def generation_enabled(self) -> bool:
        return bool(self.openai_api_key)

    @classmethod
    def from_sources(
        cls,
        environ: Mapping[str, str],
        secrets: Mapping[str, object] | None = None,
    ) -> "Settings":
        api_key = _value("OPENAI_API_KEY", environ, secrets, "") or None
        return cls(
            openai_api_key=api_key,
            openai_model=_value("OPENAI_MODEL", environ, secrets, "gpt-4.1-mini") or "gpt-4.1-mini",
            index_path=Path(_value("KNOWLEDGE_INDEX_PATH", environ, secrets, "data/processed/chunks.jsonl")),
            manifest_path=Path(_value("KNOWLEDGE_MANIFEST_PATH", environ, secrets, "data/processed/index_manifest.json")),
            chroma_persist_directory=Path(_value("CHROMA_PERSIST_DIRECTORY", environ, secrets, "data/processed/chroma")),
            grounding_validation_enabled=_boolean(
                "ENABLE_GROUNDING_VALIDATION", environ, secrets, False
            ),
            max_question_chars=_integer("MAX_QUESTION_CHARS", environ, secrets, 1000, maximum=1000),
            max_evidence_chunks=_integer("MAX_EVIDENCE_CHUNKS", environ, secrets, 5, maximum=5),
            max_evidence_chars=_integer("MAX_EVIDENCE_CHARS", environ, secrets, 6000, maximum=6000),
            session_requests=_integer("SESSION_REQUESTS", environ, secrets, 10),
            session_window_seconds=_integer("SESSION_WINDOW_SECONDS", environ, secrets, 60),
            daily_generation_limit=_integer("DAILY_GENERATION_LIMIT", environ, secrets, 500),
        )
