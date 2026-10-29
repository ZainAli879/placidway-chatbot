"""Explainable lexical retrieval that returns source evidence, not generated answers."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
# Common words add little retrieval value and can cause unrelated matches.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    return [
        token.casefold()
        for token in TOKEN_RE.findall(text)
        if token.casefold() not in STOPWORDS
    ]


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    source_id: str
    page_type: str
    source_url: str
    title: str
    text: str
    score: float
    retrieval_sources: tuple[str, ...] = ()


class KnowledgeSearch:
    def __init__(self, chunks: Iterable[dict]):
        self.chunks = list(chunks)
        self.tokens = [tokenize(item.get("text", "")) for item in self.chunks]
        self.document_frequency = Counter()
        for tokens in self.tokens:
            self.document_frequency.update(set(tokens))
        self.average_length = (
            sum(map(len, self.tokens)) / len(self.tokens) if self.tokens else 0.0
        )

    @classmethod
    def from_jsonl(cls, path="data/processed/chunks.jsonl"):
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError("Run python -m app.knowledge.indexer first.")
        rows = [
            json.loads(line)
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(rows)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks or top_k < 1:
            return []

        total, k1, b = len(self.chunks), 1.5, 0.75
        query_counts = Counter(query_tokens)
        scored = []
        for index, tokens in enumerate(self.tokens):
            counts, length, score = Counter(tokens), len(tokens), 0.0
            for term, query_count in query_counts.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency.get(term, 0)
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (
                    1 - b + b * (length / self.average_length if self.average_length else 0)
                )
                score += (
                    idf
                    * (frequency * (k1 + 1) / denominator)
                    * min(query_count, 2)
                )
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda pair: (-pair[0], str(self.chunks[pair[1]]["chunk_id"])))
        return [
            SearchResult(
                self.chunks[index]["chunk_id"],
                self.chunks[index]["source_id"],
                self.chunks[index]["page_type"],
                self.chunks[index]["source_url"],
                self.chunks[index]["title"],
                self.chunks[index]["text"],
                round(score, 4),
            )
            for score, index in scored[:top_k]
        ]
