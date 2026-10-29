"""Typed records used by the source-grounded knowledge pipeline."""
from dataclasses import asdict, dataclass
from typing import Any

@dataclass(frozen=True)
class SourceDocument:
    source_id: str
    page_type: str
    url: str
    title: str
    fetched_at_utc: str
    text: str

@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source_id: str
    page_type: str
    source_url: str
    title: str
    chunk_index: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class PriceMention:
    source_id: str
    page_type: str
    source_url: str
    amount: str
    currency: str
    context: str
    review_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
