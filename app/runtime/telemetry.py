"""Privacy-safe JSON events for stdout-based free-tier observability."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import sys
from typing import TextIO


@dataclass(frozen=True)
class RuntimeEvent:
    event: str
    response_mode: str
    latency_ms: float
    retrieval_ms: float
    generation_ms: float
    evidence_count: int
    reason: str
    exception_class: str | None

    @classmethod
    def failure(cls, reason: str, error: Exception) -> "RuntimeEvent":
        return cls("request_failed", "error", 0.0, 0.0, 0.0, 0, reason, error.__class__.__name__)


def emit_event(event: RuntimeEvent, sink: TextIO = sys.stdout) -> None:
    sink.write(json.dumps(asdict(event), separators=(",", ":")) + "\n")
    sink.flush()
