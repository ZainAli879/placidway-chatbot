# PlacidWay Knowledge Chatbot

A source-grounded chatbot project for the PlacidWay AI Automation Engineer assessment. It is designed to answer only from the seven approved PlacidWay pages, cite evidence, and avoid inventing medical or pricing information.

## Current status

- [x] Explicit allowlist of seven assessment URLs
- [x] Polite crawler with descriptive User-Agent, robots.txt check, and minimum one-second request interval
- [x] Extraction audit report with HTTP status, final URL, title, text length, and review flags
- [x] Source-aware chunking with URL and page-type metadata
- [x] Candidate price extraction written to a review-only file
- [x] Hybrid BM25 and local Chroma semantic retrieval fused with reciprocal-rank fusion (RRF)
- [x] FlashRank reranking with a dynamic three/four/five-chunk evidence budget
- [x] Structured claim-level grounding and exact source links
- [x] Reviewed artifact digest, health checks, rate/concurrency controls, and safe telemetry
- [x] Evidence-grounded OpenAI answer generation (optional; requires API key)
- [x] Package-specific evidence filtering to reduce generic comparison-page contamination
- [x] Scope fallbacks for booking, weather, personal medical advice, and insufficient evidence
- [x] Expanded evidence-first prompting for price claims, package specificity, medical safety, and citation hygiene
- [x] Separates base inclusions, extra-cost/conditional services, and optional add-ons in package answers
- [x] Streamlit chat UI and automated tests
- [ ] Manual review of extracted page text and candidate price mentions
- [ ] Run the full eight-question assessment against freshly indexed sources
- [ ] Approve the freshly rebuilt artifact after manual source review
- [ ] Deploy to Streamlit Community Cloud and monitor availability

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open the local `.env` file and set `OPENAI_API_KEY` to your own key. Optionally set `OPENAI_MODEL` (defaults to `gpt-4.1-mini`). Never paste your key into source code or commit the `.env` file.

## 1. Audit the approved pages

```powershell
python -m scripts.audit_sources
```

The crawler checks robots.txt before requesting the approved pages. It does not follow links or expand the allowlist. Successful source text is written to `data/raw/`; the report is `data/source_audit.json`.

A successful HTTP response is not proof of complete or semantically correct extraction. Review each audit row and the corresponding text before using it as chatbot evidence.

## 2. Build the local knowledge and vector indexes

```powershell
python -m app.knowledge.indexer
python -m scripts.build_vector_index --corpus data\processed\chunks.jsonl --output data\processed\chroma
```

Outputs:
- `data/processed/chunks.jsonl`: chunks with source IDs, URLs, titles, and page types.
- `data/processed/price_mentions_review.jsonl`: currency mentions with surrounding source context.
- `data/processed/index_manifest.json`: build counts and timestamp.
- `data/processed/chroma/`: persistent semantic vector index bound to the exact chunks file.

The first vector-index build downloads Chroma's approximately 80 MB ONNX MiniLM embedding model. Build it once on your own machine after installing the requirements; visitor requests never rebuild document embeddings. Rebuild the Chroma index whenever `chunks.jsonl` changes, because startup deliberately rejects a missing or stale vector artifact.

**Price safety:** candidate prices are not verified quotes. Each mention is marked `review_required: true`. Do not merge general comparison-page prices with ITC treatment-package prices. Confirm the condition, outpatient/inpatient program, duration, inclusions, exclusions, currency, and qualifiers against the original page before using a price in a user-facing answer.

## 3. Run tests and launch the chatbot

```powershell
python -m pytest
streamlit run streamlit_app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser. The app uses the local `data/processed/chunks.jsonl` index, retains conversation history for the current session, and displays source links.

If `OPENAI_API_KEY` is set, OpenAI generates concise structured claims from the top three focused, top four list-style, or top five comparison/multipart chunks selected by FlashRank. Candidates come from both BM25 keyword search and Chroma semantic search, then are deduplicated and fused with RRF before reranking. Every factual claim must cite accepted chunk IDs. If generation fails, the app reports the underlying safe error category instead of constructing an ungrounded answer.

Run the reviewed retrieval regression cases after building the vector artifact:

```powershell
python -m scripts.evaluate_retrieval --mode bm25
python -m scripts.evaluate_retrieval --mode hybrid
```

## Production artifact approval

Rebuilding always invalidates approval. Review all seven raw source files and the price-candidate report, then deliberately approve the exact corpus bytes:

```powershell
python -m app.knowledge.indexer
python -m scripts.build_vector_index --corpus data\processed\chunks.jsonl --output data\processed\chroma
python -m scripts.review_artifact --reviewer "your-name-or-email"
python -m scripts.validate_artifact
```

Commit `data/processed/chunks.jsonl`, `data/processed/index_manifest.json`, and the matching `data/processed/chroma/` artifact after approval. Never commit `data/raw`, `.env`, or chat logs. Roll back the reviewed chunks, manifest, and Chroma artifact together.

## Free Streamlit deployment

1. Push the repository to GitHub.
2. Confirm GitHub Actions passes, including artifact validation and security checks.
3. In Streamlit Community Cloud, select `streamlit_app.py` as the entry point.
4. Add `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in Streamlit Secrets.
5. Set an OpenAI project usage budget; in-process limits reset when a free instance restarts.

The app uses local Chroma ONNX embeddings and a local FlashRank TinyBERT reranker. Both models must exist in the deployment cache or may download on first startup, so a sleeping free Streamlit app can have a platform cold start. Warm retrieval and reranking are local and do not make an API call. Warm local retrieval is benchmarked with:

```powershell
python -m scripts.benchmark_runtime --iterations 10
```

The UI and JSON logs report retrieval, generation, and total latency without recording visitor questions or evidence text. Generation is not cancelled for latency, so external network/provider delays can exceed the 3–4 second warm target.

## Source scope and safety

The crawler is restricted to the seven pages in `app/config/sources.py`. Do not add automatically discovered links or fetch pages outside the assessment allowlist. The model receives only retrieved chunks, is instructed not to use outside knowledge, and cannot book appointments or provide personal medical advice. Model instructions are not a mathematical guarantee; review generated answers and test edge cases.

## Privacy and secrets

Do not commit `.env`, API keys, user chat logs, or personal/medical data. Use `.env.example` only as a template.

## Known limitations

- HTML extraction can include navigation or repeated boilerplate and may miss dynamically loaded or embedded video content.
- Price regex extraction produces candidate mentions, not validated package records.
- BM25 and Chroma each retrieve candidates, RRF combines them, and local TinyBERT selects the final chunks. This improves recall but adds local CPU work and memory compared with BM25 alone.
- Semantic recall depends on the embedding model and corpus quality. Maintain reviewed retrieval cases and inspect failures before changing chunking or ranking weights.
- Live model answers can vary; use the automated tests and manually review all eight required assessment categories before submission.
