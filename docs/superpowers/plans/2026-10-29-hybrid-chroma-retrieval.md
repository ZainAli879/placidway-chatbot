# Hybrid Chroma Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Combine BM25 and persistent local Chroma retrieval with RRF, then FlashRank and dynamic top 3–5 grounded evidence selection.

**Architecture:** A new hybrid searcher runs lexical and dense retrieval independently and fuses ranks while preserving retrieval provenance. An offline command creates a digest-bound Chroma artifact; Streamlit validates and loads it without document embedding at request time.

**Tech Stack:** Python 3.12, ChromaDB, ONNX MiniLM embeddings, FlashRank, pytest, Streamlit

**Spec:** `docs/superpowers/specs/2026-10-29-hybrid-chroma-retrieval-design.md`

## Global Constraints

- Keep all existing grounding, citation, package-isolation, safety, greeting, booking, and unrelated-query behavior.
- Never rebuild document embeddings during a visitor request.
- Do not silently fall back to lexical-only retrieval when the deployed vector artifact is missing or stale.
- Do not log queries, evidence, secrets, or exception messages.
- Maximum generation evidence remains 6,000 characters and five chunks.

## Review Focus

- Dense-only matches with zero lexical overlap must survive into reranking.
- A named package query must never receive treatment chunks from another package.
- RRF must deduplicate chunks found by both retrieval paths.
- Stale vector artifacts must fail startup before answering.
- Focused/list/complex questions must pass exactly 3/4/5 maximum chunks respectively.

---

### Task 1: Hybrid retrieval and RRF

**Files:**
- Create: `app/retrieval/hybrid_search.py`
- Modify: `app/retrieval/search.py`
- Create: `tests/test_hybrid_search.py`

**Interfaces:**
- Produces: `HybridKnowledgeSearch.search(query: str, *, top_k: int) -> list[SearchResult]` and `SearchResult.retrieval_sources: tuple[str, ...]`.

- [ ] Write failing tests for lexical-only, vector-only, duplicate, deterministic RRF ordering, and empty-query behavior.
- [ ] Run focused tests and verify failure because the hybrid searcher/provenance do not exist.
- [ ] Implement RRF fusion with injected collection support for unit tests.
- [ ] Run focused tests and verify they pass.

### Task 2: Digest-bound persistent vector artifact

**Files:**
- Modify: `app/retrieval/hybrid_search.py`
- Create: `scripts/build_vector_index.py`
- Create: `tests/test_vector_artifact.py`

**Interfaces:**
- Produces: `build_vector_index(corpus_path, persist_directory, collection_name, embedding_function)` and runtime validation of corpus digest, count, IDs, and model identifier.

- [ ] Write failing tests for valid, missing, and stale vector artifacts.
- [ ] Run focused tests and verify the validation behavior is absent.
- [ ] Implement offline collection creation and fail-closed runtime validation.
- [ ] Run focused tests and verify they pass.

### Task 3: Engine and Streamlit integration

**Files:**
- Modify: `app/answering/engine.py`
- Modify: `app/answering/openai_generator.py`
- Modify: `app/retrieval/reranker.py`
- Modify: `app/config/settings.py`
- Modify: `streamlit_app.py`
- Modify: `.env.example`
- Modify: `tests/test_list_retrieval.py`
- Modify: `tests/test_settings.py`
- Modify: `tests/test_openai_generator.py`

**Interfaces:**
- Consumes: hybrid search results with retrieval provenance.
- Produces: semantic-only acceptance and dynamic 3/4/5 evidence selection with a five-chunk generator cap.

- [ ] Write failing tests for dense-only candidate acceptance and focused/list/complex 3/4/5 selection.
- [ ] Run focused tests and verify current lexical gate and fixed limits fail them.
- [ ] Integrate hybrid search, preserve provenance during reranking, and implement evidence limits.
- [ ] Run focused and existing answer tests.

### Task 4: Dependencies, artifact, evaluation, and deployment documentation

**Files:**
- Modify: `requirements.txt`
- Modify: `requirements.lock`
- Modify: `requirements-dev.lock`
- Modify: `tests/test_production_dependencies.py`
- Create: `tests/test_retrieval_evaluation.py`
- Modify: `.gitignore`
- Modify: `README.md`
- Create: `data/processed/chroma/**`

**Interfaces:**
- Produces: deployable local Chroma artifact and regression evaluation for hybrid recall.

- [ ] Update dependency tests to require Chroma without Sentence Transformers/PyTorch.
- [ ] Install and pin compatible dependencies, then run `pip check`.
- [ ] Build the persistent vector artifact offline and verify runtime validation.
- [ ] Add retrieval evaluation fixtures and compare BM25/hybrid expected-chunk recall.
- [ ] Update documentation and ignore rules for rebuilding and deploying the vector artifact.
- [ ] Run the complete test suite, compilation, artifact validation, dependency check, and diff check.
- [ ] Commit all scoped files while preserving the user-owned `data/processed/index_manifest.json` change.

