# Production Performance and Safety Design

## Intent

Turn the current assessment prototype into a reliable, free-hosting-friendly Streamlit deployment. Normal warm requests should target three-to-four-second performance, remain grounded in the seven approved sources, and degrade safely when OpenAI or optional infrastructure is unavailable.

## Goals

- Deploy on Streamlit Community Cloud without a separate paid backend.
- Keep warm request latency at or below four seconds by enforcing a hard application budget.
- Remove runtime cross-encoder reranking and heavyweight vector-model work from the request path.
- Send at most three compact evidence chunks to OpenAI.
- Return a retrieved-evidence response when generation fails; it must be composed exclusively from accepted chunks and their source links.
- Remove package facts from model instructions; facts must come only from retrieved evidence.
- Make the reviewed knowledge index reproducible, versioned, validated, and deployable from Git.
- Add safe configuration, input limits, lightweight session throttling, structured operational events, health diagnostics, CI, dependency locking, and production documentation.
- Preserve existing booking, medical-advice, ambiguity, and insufficient-evidence safeguards.

## Non-goals

- A Vercel/Netlify rewrite or separate JavaScript frontend.
- Guaranteed sub-four-second cold starts after Streamlit has suspended the app.
- Distributed rate limiting across multiple server instances.
- Clinical decision support, diagnosis, treatment recommendations, booking, or live price verification.
- Adding a paid database, cache, monitoring service, or artifact registry.

## Anti-goals

- Do not improve latency by weakening evidence checks or silently inventing answers.
- Do not commit Chroma databases, downloaded models, secrets, or chat logs.
- Do not perform crawling, model downloads, or index construction during a visitor request.
- Do not hide degraded behavior from the UI or logs.

## Constraints

- Hosting target: Streamlit Community Cloud free tier.
- Warm response performance target: 3–4 seconds, measured and reported but not enforced with a generation timeout.
- OpenAI generation is not cancelled for latency; external API or network slowness can exceed the target.
- Maximum visitor question length: 1,000 characters.
- Maximum model evidence: three chunks and 6,000 evidence characters total.
- Runtime retrieval is deterministic lexical BM25 over the small approved corpus.
- The source corpus remains restricted to the seven explicit allowlisted PlacidWay URLs.
- No user question or medical content is written to logs.
- Python runtime: 3.12.
- Chat history retained in a session: at most 20 messages.
- Session throttle: 10 accepted questions per rolling 60 seconds.
- Process-wide concurrent generation limit: two; excess requests receive a busy response without calling OpenAI.
- Process-local daily OpenAI ceiling: 500 calls by default, configurable downward. Free hosting cannot provide a tamper-proof or cross-replica quota, so provider-side project budgets remain required.

## Approaches Considered

### 1. Keep hybrid retrieval and optimize the reranker

This preserves semantic recall, but model loading, Chroma startup, and cross-encoder inference remain unreliable on a sleeping free instance. It also carries the largest dependency and memory footprint.

### 2. Streamlit-first compact runtime (selected)

Use BM25 retrieval with package-aware filtering, compact evidence, and the existing answer safeguards. Build and validate the knowledge artifact ahead of deployment. This best matches the tiny seven-page corpus and the four-second budget.

### 3. Vercel frontend plus Python API

This provides cleaner frontend/backend separation but requires a rewrite, introduces serverless cold starts and bundle constraints, and does not solve model latency by itself.

## Architecture

### Offline knowledge build

The crawler and indexer remain offline commands. The indexer emits a compact JSONL corpus plus an unreviewed manifest containing schema version, SHA-256 of the exact corpus bytes, source IDs/count, chunk count, and build time. Rebuilding always clears approval. A separate deliberate approval command requires a non-empty reviewer identity, verifies all seven sources are represented, and writes `reviewed_by`, `reviewed_at_utc`, and the current corpus digest. Validation rejects missing sources, unknown URLs, malformed chunks, digest mismatch, or absent/stale approval. Only the compact reviewed artifact and manifest are deployed; raw crawl output, Chroma state, and downloaded models remain ignored.

### Runtime request flow

The measured boundary begins immediately when the application accepts a submitted question and ends when an `Answer` object is ready for Streamlit to render. The target is no more than 4.0 seconds with 0.1 seconds of test tolerance; Streamlit network/render time and platform cold-start time are reported separately.

1. Start monotonic latency measurement, then validate and normalize the question; reject empty or over-1,000-character input.
2. Apply a per-session sliding-window request limit.
3. Continue measuring every stage so retrieval and generation latency are reported separately.
4. Run scope, medical-safety, ambiguity, and package checks.
5. Retrieve BM25 candidates using case-folded alphanumeric tokens and deterministic score/chunk-ID ordering. For a named package, accept only that package's treatment-page chunks; never backfill another package. Otherwise require at least one meaningful query-token match, deduplicate normalized text, and prefer source diversity after relevance. Follow-ups keep only the last explicitly named package context. No candidates means abstention.
6. Select no more than three chunks and compact them to a 6,000-character evidence budget without cutting through a price token or URL.
7. Call OpenAI with zero SDK retries, a reused client, a compact prompt, and a reduced output-token ceiling. Do not impose or simulate a latency timeout; wait for the generation result and measure its duration.
8. Require structured generated output in which every factual answer sentence lists one or more accepted chunk IDs. Reject unknown chunk IDs, cross-package citations, unsupported monetary amounts, or factual sentences without citations. Disclaimer/limitation sentences may be uncited. Render claim-level source links from validated IDs.
9. If generation errors or returns invalid output, construct a retrieved-evidence response exclusively from the accepted chunks, with their source links. If those chunks cannot directly answer the question, abstain instead of filling the gap. Slowness alone does not trigger this response.
10. Emit a structured operational event containing timing, mode, reason, evidence count, and exception class, but never question/evidence text.

### Configuration

A typed `Settings` object validates environment values once. Environment variables override matching Streamlit secrets. Index path and limits are required to be valid; an absent OpenAI key is allowed and intentionally starts retrieved-evidence-only mode. An invalid non-empty key degrades on the first failed call without exposing it. Runtime hybrid mode and reranking are removed from production configuration.

### Reliability and observability

The UI exposes a small diagnostics section with index version, source count, retrieval mode, last response latency, and whether generation is enabled. Structured JSON events go to standard output for Streamlit logs. Broad exception handlers record exception type and stage before safe fallback. A health check function validates settings and artifact integrity for CI and startup. Session history is trimmed to the newest 20 messages.

### Security and privacy

Question length and request frequency are bounded. A rolling per-session limiter permits 10 questions per 60 seconds; a process-wide non-blocking semaphore permits two concurrent generation calls; a process-local daily counter defaults to 500 calls. Violations do not call OpenAI and return a clear retry/budget message. Session reset can bypass the session limiter and process restarts reset the daily counter; these are accepted free-tier residual risks, documented alongside the need for an OpenAI project budget. Source URLs are checked against the allowlist before display. Logs contain no prompt, evidence, API key, or user medical text. The `.env` and Streamlit secrets remain ignored.

Dependencies are pinned in `requirements.lock` for Python 3.12. Runtime dependencies exclude Chroma, sentence-transformers, and cross-encoders. GitHub Actions fails on tests, compilation, artifact validation, `pip check`, `pip-audit`, or `detect-secrets` findings. A source scan also fails if runtime files import `chromadb` or `sentence_transformers`.

### Retrieved-evidence response contract

When generation is disabled, late, invalid, or unavailable, the response is not a generic answer. It uses at most the three accepted chunks, selecting the highest-ranked sentences that contain meaningful query terms. It excludes sentences containing promotional efficacy/cure/safety claims unless the visitor explicitly asks what the page claims, in which case the wording is attributed to the page. It preserves price text exactly, truncates only at sentence boundaries, and supplies a source link for every excerpt. If no accepted sentence directly addresses the question, it returns an insufficiency response with no factual claim.

## Testing and acceptance

- Existing behavior tests continue to pass.
- Tests prove the reranker and Chroma are not loaded in the production request path.
- Tests prove no more than three chunks or 6,000 evidence characters reach the generator.
- Benchmarks record local retrieval latency, evidence size, generation latency, and total warm latency without cancelling a delayed generator.
- Tests prove model instructions contain policy but no package-specific facts and every accepted generated factual sentence carries valid claim-level chunk citations.
- Tests cover invalid settings, oversized input, throttling, corrupted/unreviewed artifacts, safe logs, generator errors, and prove that the retrieved-evidence response contains no factual content outside accepted chunks.
- Tests cover deterministic package-isolated retrieval, follow-up context, history trimming, the two-call concurrency ceiling, daily budget exhaustion, and no runtime heavyweight imports/downloads.
- A benchmark over representative local questions records retrieval and end-to-end timing. The four-second service target applies to warm runtime requests; free-host cold starts are reported separately and cannot be guaranteed.

## Deployment

GitHub is the source of truth. CI validates the committed knowledge artifact. Streamlit Community Cloud deploys `streamlit_app.py`, reads `OPENAI_API_KEY` from Streamlit secrets, and installs `requirements.lock`. No runtime path imports or downloads an embedding or reranker model. The README documents deployment, index review/approval/rebuild, rollback, operational checks, provider-side OpenAI budget setup, free-tier cold starts, and the fact that OpenAI API usage is not included in free hosting.

