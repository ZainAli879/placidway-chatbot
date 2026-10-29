"""Hybrid BM25 and Chroma retrieval combined with reciprocal-rank fusion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from app.retrieval.search import KnowledgeSearch, SearchResult

DEFAULT_COLLECTION_NAME = "placidway_chunks"
DEFAULT_EMBEDDING_MODEL = "chroma-default-all-MiniLM-L6-v2"


class VectorArtifactError(ValueError):
    """The persisted dense index does not match the reviewed chunk corpus."""


def _read_corpus(corpus_path: str | Path) -> tuple[Path, list[dict], str]:
    path = Path(corpus_path)
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise VectorArtifactError(f"Vector corpus cannot be loaded: {exc.__class__.__name__}") from exc
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, rows, digest


def validate_vector_collection(
    collection,
    chunks: list[dict],
    corpus_sha256: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> None:
    metadata = collection.metadata or {}
    if metadata.get("corpus_sha256") != corpus_sha256:
        raise VectorArtifactError("Vector artifact digest does not match the reviewed corpus")
    if metadata.get("embedding_model") != embedding_model:
        raise VectorArtifactError("Vector artifact embedding model is unsupported")
    if int(metadata.get("chunk_count", -1)) != len(chunks) or collection.count() != len(chunks):
        raise VectorArtifactError("Vector artifact chunk count does not match the reviewed corpus")
    expected_ids = {str(row["chunk_id"]) for row in chunks}
    actual_ids = {str(value) for value in collection.get(include=[]).get("ids", [])}
    if actual_ids != expected_ids:
        raise VectorArtifactError("Vector artifact chunk IDs do not match the reviewed corpus")


def build_vector_index(
    corpus_path: str | Path,
    *,
    persist_directory: str | Path = "data/processed/chroma",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    client=None,
    embedding_function=None,
):
    """Recreate the persistent Chroma collection outside the visitor request path."""
    _, chunks, digest = _read_corpus(corpus_path)
    if client is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(persist_directory))
    if embedding_function is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        embedding_function = DefaultEmbeddingFunction()
    try:
        client.delete_collection(collection_name)
    except Exception:
        # A missing collection is the normal first-build case.
        pass
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={
            "corpus_sha256": digest,
            "embedding_model": embedding_model,
            "chunk_count": len(chunks),
            "hnsw:space": "cosine",
        },
    )
    if chunks:
        collection.upsert(
            ids=[str(row["chunk_id"]) for row in chunks],
            documents=[f"{row.get('title', '')}\n\n{row.get('text', '')}" for row in chunks],
            metadatas=[{
                "source_id": str(row["source_id"]),
                "page_type": str(row["page_type"]),
                "source_url": str(row["source_url"]),
                "title": str(row["title"]),
            } for row in chunks],
        )
    validate_vector_collection(collection, chunks, digest, embedding_model)
    return collection


class HybridKnowledgeSearch:
    """Fuse independent lexical and dense rankings without mixing raw scores."""

    def __init__(
        self,
        chunks: Iterable[dict],
        *,
        collection,
        lexical_search: KnowledgeSearch | None = None,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        self.chunks = list(chunks)
        self.by_id = {str(row["chunk_id"]): row for row in self.chunks}
        self.lexical = lexical_search or KnowledgeSearch(self.chunks)
        self.collection = collection
        self.rrf_k = rrf_k

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path = "data/processed/chunks.jsonl",
        *,
        persist_directory: str | Path = "data/processed/chroma",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> "HybridKnowledgeSearch":
        _, chunks, digest = _read_corpus(path)
        try:
            import chromadb
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            client = chromadb.PersistentClient(path=str(persist_directory))
            collection = client.get_collection(
                name=collection_name,
                embedding_function=DefaultEmbeddingFunction(),
            )
        except Exception as exc:
            raise VectorArtifactError(f"Vector artifact cannot be opened: {exc.__class__.__name__}") from exc
        validate_vector_collection(collection, chunks, digest, embedding_model)
        return cls(chunks, collection=collection)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        if not query.strip() or top_k < 1 or not self.chunks:
            return []

        candidate_k = min(max(top_k, 1), len(self.chunks))
        lexical = self.lexical.search(query, top_k=candidate_k)
        vector = self.collection.query(
            query_texts=[query],
            n_results=candidate_k,
            include=["metadatas", "distances"],
        )

        scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}
        results: dict[str, SearchResult] = {}
        for rank, item in enumerate(lexical, start=1):
            results[item.chunk_id] = item
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            sources.setdefault(item.chunk_id, set()).add("lexical")

        ids = (vector.get("ids") or [[]])[0]
        for rank, chunk_id in enumerate(ids, start=1):
            chunk_id = str(chunk_id)
            row = self.by_id.get(chunk_id)
            if row is None:
                continue
            if chunk_id not in results:
                results[chunk_id] = SearchResult(
                    chunk_id=chunk_id,
                    source_id=str(row["source_id"]),
                    page_type=str(row["page_type"]),
                    source_url=str(row["source_url"]),
                    title=str(row["title"]),
                    text=str(row["text"]),
                    score=0.0,
                )
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            sources.setdefault(chunk_id, set()).add("vector")

        ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
        fused = []
        for chunk_id in ordered[:top_k]:
            item = results[chunk_id]
            fused.append(SearchResult(
                item.chunk_id,
                item.source_id,
                item.page_type,
                item.source_url,
                item.title,
                item.text,
                round(scores[chunk_id], 8),
                tuple(name for name in ("lexical", "vector") if name in sources[chunk_id]),
            ))
        return fused
