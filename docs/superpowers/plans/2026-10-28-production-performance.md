# Production Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit Community Cloud-ready chatbot whose warm path uses compact deterministic retrieval, grounded generation, safe retrieved-evidence degradation, and measurable 3–4 second target performance without cancelling slow OpenAI calls.

**Architecture:** Replace runtime hybrid/vector/reranker work with cached BM25 over a reviewed, digest-validated JSONL artifact. Route each validated request through deterministic package isolation, a three-chunk/6,000-character evidence budget, structured claim citations, lightweight controls, and privacy-safe telemetry. OpenAI remains optional and un-timed; failures use accepted retrieved evidence only.

**Tech Stack:** Python 3.12, Streamlit, OpenAI Responses API, pytest, GitHub Actions, JSONL/SHA-256 artifacts.

**Spec:** `docs/superpowers/specs/2026-10-28-production-performance-design.md`

## Global Constraints

- Hosting target is Streamlit Community Cloud free tier.
- Warm response target is 3–4 seconds, measured but not enforced by cancelling generation.
- Runtime retrieval is lexical BM25 only; no Chroma, sentence-transformer, or cross-encoder import/download.
- Generator input is at most three accepted chunks and 6,000 evidence characters.
- Every generated factual sentence must cite accepted chunk IDs; unsupported output degrades to retrieved evidence.
- Questions are at most 1,000 characters; session history is at most 20 messages.
- Session rate is 10 accepted questions per rolling 60 seconds; generation concurrency is two; default process-local daily generation ceiling is 500.
- Logs must not contain questions, evidence, API keys, or medical text.
- No factual package knowledge may be embedded in model instructions.

## Review Focus

- Named-package follow-ups must never mix evidence from another package; Task 4 tests exact package isolation and missing-package abstention.
- Structured output with unknown/missing chunk IDs must never render as grounded; Task 3 tests rejection and evidence fallback.
- Rebuilt or edited corpus bytes must invalidate approval; Task 2 tests digest and review invalidation.
- Concurrent or rate-limited requests must not consume OpenAI quota; Task 5 tests limiter and semaphore behavior.
- No-key, invalid-key, and provider-error modes must remain useful without leaking content; Tasks 1, 3, and 6 test these paths.

---

### Task 1: Validated Runtime Settings

**Files:**
- Create: `app/config/settings.py`
- Create: `tests/test_settings.py`
- Modify: `.env.example`

**Interfaces:**
- Produces: `Settings.from_sources(environ: Mapping[str, str], secrets: Mapping[str, object] | None = None) -> Settings`
- Produces fields: `openai_api_key`, `openai_model`, `index_path`, `manifest_path`, `max_question_chars`, `max_evidence_chunks`, `max_evidence_chars`, `session_requests`, `session_window_seconds`, `generation_concurrency`, `daily_generation_limit`.

- [ ] **Step 1: Write failing settings tests**

Cover environment-over-secret precedence, optional empty API key, integer bounds, fixed production maxima (1,000/3/6,000), and secret-safe validation errors.

- [ ] **Step 2: Run `python -m pytest tests/test_settings.py -q` and verify the expected import/behavior failures**

- [ ] **Step 3: Implement immutable validated `Settings` and document every variable in `.env.example`**

- [ ] **Step 4: Run the focused tests and full suite**

- [ ] **Step 5: Commit `feat: add validated production settings`**

### Task 2: Reproducible Reviewed Knowledge Artifact

**Files:**
- Create: `app/knowledge/artifact.py`
- Create: `scripts/review_artifact.py`
- Create: `scripts/validate_artifact.py`
- Create: `tests/test_artifact.py`
- Modify: `app/knowledge/indexer.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `build_manifest(corpus_path: Path, chunks: Sequence[dict]) -> dict`
- Produces: `approve_artifact(corpus_path: Path, manifest_path: Path, reviewer: str, now: datetime) -> dict`
- Produces: `validate_artifact(corpus_path: Path, manifest_path: Path) -> ArtifactMetadata`
- Consumes: `APPROVED_SOURCES` as the only allowed source IDs and URLs.

- [ ] **Step 1: Write failing artifact lifecycle tests**

Test seven-source completeness, exact-byte SHA-256, unknown URL rejection, non-empty reviewer identity, approval invalidation after corpus mutation/rebuild, and malformed JSONL rejection.

- [ ] **Step 2: Run `python -m pytest tests/test_artifact.py -q` and verify failures are caused by missing interfaces**

- [ ] **Step 3: Implement manifest creation, deliberate approval, validation, and CLI exit codes**

Indexer writes an unreviewed manifest every build. Approval writes reviewer/time only after validation. Do not approve the current corpus automatically.

- [ ] **Step 4: Update ignore rules so raw crawl, Chroma, and review candidates remain ignored while reviewed `chunks.jsonl` and `index_manifest.json` can be committed**

- [ ] **Step 5: Run focused tests, build a temporary fixture artifact, validate it, then run the full suite**

- [ ] **Step 6: Commit `feat: add reviewed knowledge artifact lifecycle`**

### Task 3: Compact Grounded Generator

**Files:**
- Modify: `app/answering/openai_generator.py`
- Create: `app/answering/grounding.py`
- Modify: `tests/test_openai_generator.py`
- Create: `tests/test_grounding.py`

**Interfaces:**
- Produces: `GeneratedClaim(text: str, chunk_ids: tuple[str, ...])`
- Produces: `GroundedGeneration(claims: tuple[GeneratedClaim, ...])`
- Produces: `validate_generation(payload: object, evidence: Sequence[SearchResult], package_key: str | None) -> str | None`
- Changes: `OpenAIAnswerGenerator.generate(question, evidence, package_key=None) -> str`; sends compact structured-output instructions, `max_output_tokens=250`, and uses an OpenAI client configured with `max_retries=0` but no latency timeout.

- [ ] **Step 1: Write failing tests for compact input and grounded output**

Prove three-chunk/6,000-character caps, no package facts in instructions, valid claim rendering with claim-level links, rejection of uncited/unknown/cross-package claims, exact price support, and provider errors propagating to the engine.

- [ ] **Step 2: Run the generator/grounding tests and verify expected failures**

- [ ] **Step 3: Implement structured generation and deterministic validation**

Factual sentences require accepted chunk IDs; limitation/disclaimer sentences may be uncited. Do not use background threads or timeout-based fallback.

- [ ] **Step 4: Run focused tests and full suite**

- [ ] **Step 5: Commit `feat: enforce claim-level grounding`**

### Task 4: Fast Deterministic Retrieval and Evidence Response

**Files:**
- Modify: `app/retrieval/search.py`
- Create: `app/answering/evidence_response.py`
- Modify: `app/answering/engine.py`
- Modify: `tests/test_search.py`
- Modify: `tests/test_answer_engine.py`
- Modify: `tests/test_list_retrieval.py`
- Modify: `tests/test_intent_context_separation.py`

**Interfaces:**
- Produces: `select_evidence(question: str, candidates: Sequence[SearchResult], package_key: str | None, max_chunks=3, max_chars=6000) -> tuple[SearchResult, ...]`
- Produces: `build_evidence_response(question: str, evidence: Sequence[SearchResult]) -> str | None`
- Changes: `Answer` gains `latency_ms`, `retrieval_ms`, `generation_ms`, and `response_mode`.

- [ ] **Step 1: Write failing retrieval and engine tests**

Cover deterministic ties, maximum evidence caps, named-package isolation, follow-up preservation, no relevant sentence abstention, promotional safety filtering, generator error evidence response, and no-generator evidence response.

- [ ] **Step 2: Run focused tests and verify they fail for missing limits/isolation**

- [ ] **Step 3: Implement deterministic selection and sentence-boundary retrieved-evidence responses**

Remove eight-chunk list expansion. Preserve existing safety/ambiguity gates. A generation error may only return accepted evidence or abstain.

- [ ] **Step 4: Remove production usage of `HybridKnowledgeSearch`; retain legacy module only until Task 8 deletion check**

- [ ] **Step 5: Run focused tests and full suite**

- [ ] **Step 6: Commit `perf: use compact deterministic evidence retrieval`**

### Task 5: Rate, Concurrency, History, and Cost Controls

**Files:**
- Create: `app/runtime/controls.py`
- Create: `app/runtime/__init__.py`
- Create: `tests/test_runtime_controls.py`
- Modify: `streamlit_app.py`

**Interfaces:**
- Produces: `SlidingWindowLimiter.allow(now: float) -> tuple[bool, float]`
- Produces: `GenerationGate.try_acquire(now: datetime) -> ContextManager[bool]`
- Produces: `trim_messages(messages: Sequence[dict], limit: int = 20) -> list[dict]`

- [ ] **Step 1: Write failing deterministic-clock tests**

Test 10/60-second boundary behavior, two-call concurrency, daily ceiling, no counter increment when denied, and 20-message trimming.

- [ ] **Step 2: Run focused tests and verify expected failures**

- [ ] **Step 3: Implement thread-safe in-process controls and integrate them before generation**

Denied requests never call the generator. Keep session-reset and process-reset limitations documented.

- [ ] **Step 4: Run focused tests and full suite**

- [ ] **Step 5: Commit `feat: add lightweight abuse and cost controls`**

### Task 6: Privacy-Safe Telemetry, Health, and Streamlit UX

**Files:**
- Create: `app/runtime/telemetry.py`
- Create: `app/runtime/health.py`
- Create: `tests/test_telemetry.py`
- Create: `tests/test_health.py`
- Modify: `streamlit_app.py`

**Interfaces:**
- Produces: `emit_event(event: RuntimeEvent, sink: TextIO = sys.stdout) -> None`
- Produces: `check_health(settings: Settings) -> HealthReport`
- UI consumes `Answer` timings and artifact metadata.

- [ ] **Step 1: Write failing telemetry and health tests**

Assert JSON fields, absence of question/evidence/key content, exception-class-only errors, artifact/settings failure reports, and valid ready state.

- [ ] **Step 2: Run focused tests and verify failures**

- [ ] **Step 3: Implement telemetry/health and integrate cached startup plus diagnostics**

Load only validated BM25 artifact in `st.cache_resource`; remove hybrid selector, model-download messaging, mojibake strings, and silent exceptions.

- [ ] **Step 4: Run focused tests, import/compile checks, and full suite**

- [ ] **Step 5: Commit `feat: add production health and telemetry`**

### Task 7: Performance Benchmark and Regression Budget

**Files:**
- Create: `scripts/benchmark_runtime.py`
- Create: `tests/test_performance_contract.py`

**Interfaces:**
- Produces CLI JSON containing `retrieval_p50_ms`, `retrieval_p95_ms`, evidence counts/chars, optional generation/total samples, and target status.

- [ ] **Step 1: Write failing contract tests using the real committed small corpus or a representative fixture**

Assert local warm retrieval stays below 100 ms per representative query, evidence never exceeds limits, and delayed generation is measured rather than cancelled.

- [ ] **Step 2: Run focused tests and verify failures**

- [ ] **Step 3: Implement benchmark with warm-up and monotonic timing**

- [ ] **Step 4: Run benchmark in lexical/no-key mode and store only the command/results summary in documentation, not machine-specific generated files**

- [ ] **Step 5: Run focused tests and full suite**

- [ ] **Step 6: Commit `test: add runtime performance contract`**

### Task 8: Lightweight Deployment, CI, and Documentation

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.streamlit/config.toml`
- Create: `requirements.lock`
- Create: `requirements-dev.lock`
- Modify: `requirements.txt`
- Modify: `README.md`
- Modify: `.gitignore`
- Delete: `app/retrieval/hybrid_search.py`
- Delete: `tests/test_hybrid_search.py`

**Interfaces:**
- CI contract: Python 3.12 installs exact pins, runs `pytest`, `compileall`, artifact validation, `pip check`, `pip-audit`, and `detect-secrets scan` with non-zero failure on findings.
- Deployment contract: `streamlit run streamlit_app.py` needs no runtime model download or mutable vector store.

- [ ] **Step 1: Add a failing production dependency/import test**

It imports the application in a controlled no-key environment and fails if Chroma, sentence-transformers, or cross-encoder modules are required or runtime dependencies list them.

- [ ] **Step 2: Run the test and verify it fails against current hybrid dependencies**

- [ ] **Step 3: Pin Python 3.12-compatible runtime and dev dependencies; remove vector/reranker packages and legacy files**

- [ ] **Step 4: Add CI and Streamlit configuration**

- [ ] **Step 5: Rewrite README sections for review/approval, free deployment, secrets, OpenAI project budgets, cold starts, rollback, diagnostics, and benchmark commands**

- [ ] **Step 6: Run exact clean-install-equivalent checks, full tests, compile, artifact validation, security scans, benchmark, and `git diff --check`**

- [ ] **Step 7: Review every approved-design requirement against the diff and document any unavoidable external limitation**

- [ ] **Step 8: Commit `chore: prepare Streamlit production deployment`**

