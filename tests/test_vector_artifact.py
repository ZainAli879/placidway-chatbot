import hashlib
import json

import pytest

from app.retrieval.hybrid_search import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    VectorArtifactError,
    build_vector_index,
    validate_vector_collection,
)


def write_corpus(path):
    rows = [{
        "chunk_id": "c1",
        "source_id": "source-a",
        "page_type": "profile",
        "source_url": "https://example.com/a",
        "title": "Title",
        "text": "Document text",
    }]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


class FakeCollection:
    def __init__(self, metadata, ids=("c1",)):
        self.metadata = metadata
        self._ids = list(ids)
        self.upserted = None

    def count(self):
        return len(self._ids)

    def get(self, include):
        return {"ids": self._ids}

    def upsert(self, **kwargs):
        self.upserted = kwargs
        self._ids = list(kwargs["ids"])


def test_vector_collection_accepts_exact_corpus_digest_ids_and_model(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    rows = write_corpus(corpus)
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    collection = FakeCollection({
        "corpus_sha256": digest,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_count": 1,
    })

    validate_vector_collection(collection, rows, digest, DEFAULT_EMBEDDING_MODEL)


@pytest.mark.parametrize("field,value", [
    ("corpus_sha256", "stale"),
    ("embedding_model", "wrong-model"),
    ("chunk_count", 2),
])
def test_vector_collection_rejects_stale_metadata(tmp_path, field, value):
    corpus = tmp_path / "chunks.jsonl"
    rows = write_corpus(corpus)
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    metadata = {
        "corpus_sha256": digest,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_count": 1,
    }
    metadata[field] = value

    with pytest.raises(VectorArtifactError):
        validate_vector_collection(FakeCollection(metadata), rows, digest, DEFAULT_EMBEDDING_MODEL)


def test_vector_collection_rejects_mismatched_chunk_ids(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    rows = write_corpus(corpus)
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    collection = FakeCollection({
        "corpus_sha256": digest,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_count": 1,
    }, ids=("other",))

    with pytest.raises(VectorArtifactError):
        validate_vector_collection(collection, rows, digest, DEFAULT_EMBEDDING_MODEL)


def test_offline_builder_recreates_and_populates_collection(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    write_corpus(corpus)

    class FakeClient:
        def __init__(self):
            self.deleted = []
            self.collection = None

        def delete_collection(self, name):
            self.deleted.append(name)

        def get_or_create_collection(self, **kwargs):
            self.collection = FakeCollection(kwargs["metadata"], ids=())
            return self.collection

    client = FakeClient()
    collection = build_vector_index(corpus, client=client, embedding_function=object())

    assert client.deleted == [DEFAULT_COLLECTION_NAME]
    assert collection.upserted["ids"] == ["c1"]
    assert collection.upserted["documents"] == ["Title\n\nDocument text"]
    assert collection.metadata["corpus_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
