<div align="center">
  <img src="assets/placidway-logo.png" alt="PlacidWay" width="260">

  # PlacidWay Knowledge Chatbot

  **A source-grounded assistant for approved PlacidWay clinic and treatment-package information.**

  [![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![Artifact](https://img.shields.io/badge/artifact-reviewed-16A34A)](#verification)
  [![Grounded RAG](https://img.shields.io/badge/RAG-hybrid%20retrieval-F05A55)](#architecture)
</div>

## Overview

The PlacidWay Knowledge Chatbot helps visitors find information from seven approved PlacidWay pages. It answers questions about listed treatment packages, prices, durations, inclusions, comparisons, and the ITC clinic.

The system prioritizes accuracy and honesty. It retrieves relevant website evidence before generating an answer, validates factual claims against that evidence, cites the supporting PlacidWay pages, and clearly says when the approved content does not provide enough information.

## Key features

- Explicit allowlist of seven approved PlacidWay pages
- Hybrid keyword and semantic retrieval
- Reciprocal-rank fusion and local FlashRank reranking
- Dynamic selection of the strongest 3–5 evidence chunks
- Structured OpenAI responses with chunk-level grounding validation
- Clean, deduplicated links to supporting PlacidWay pages
- Follow-up questions with session-local conversation context
- Professional handling of greetings and insufficient information
- Safe responses for booking, weather, and personal medical-advice requests
- Reviewed and integrity-checked knowledge artifacts
- Responsive PlacidWay-branded chat interface
- Startup health checks and reviewed-artifact validation

## Architecture

```text
Visitor question
       │
       ▼
Intent and follow-up context resolution
       │
       ▼
BM25 keyword retrieval ──┐
                        ├── Reciprocal-rank fusion
Chroma semantic search ──┘
       │
       ▼
FlashRank reranking
       │
       ▼
Top 3–5 reviewed evidence chunks
       │
       ▼
Structured OpenAI answer generation
       │
       ▼
Grounding validation and trusted source rendering
       │
       ▼
Answer with clickable PlacidWay sources
```

Document embeddings and reranking run locally. OpenAI receives the visitor question and only the selected evidence excerpts needed to generate the answer.

## Technology choices

| Layer | Technology | Purpose |
|---|---|---|
| User interface | Streamlit | Responsive conversational interface |
| Answer generation | OpenAI Responses API | Structured, evidence-based answer synthesis |
| Default model | `gpt-4.1-mini` | Fast, cost-conscious grounded generation |
| Keyword retrieval | BM25-style local search | Exact term, package, and price matching |
| Semantic retrieval | Chroma + MiniLM ONNX | Meaning-based retrieval |
| Rank fusion | Reciprocal-rank fusion | Combines independent retrieval rankings |
| Reranking | FlashRank TinyBERT | Selects the strongest final evidence |
| Crawling | Requests + Beautiful Soup + lxml | Polite extraction from approved pages |
| Quality checks | Artifact validation and retrieval evaluation | Corpus integrity and retrieval-quality checks |

## Knowledge scope

The assistant is restricted to the seven sources registered in `app/config/sources.py`. The crawler:

- checks `robots.txt` before fetching pages;
- uses a descriptive User-Agent;
- waits at least one second between requests;
- does not follow discovered links; and
- never expands the allowlist automatically.

Each indexed chunk retains its source ID, page type, page title, URL, and stable chunk ID. This metadata supports filtering, grounding validation, and user-facing citations.

## Getting started

### Requirements

- Python 3.12
- An OpenAI API key
- Git

### 1. Clone the repository

```bash
git clone https://github.com/ZainAli879/placidway-chatbot.git
cd placidway-chatbot
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install the application dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example configuration:

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```bash
# macOS or Linux
cp .env.example .env
```

Set the required key in `.env`:

```dotenv
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
ENABLE_GROUNDING_VALIDATION=true
```

Never commit `.env`, API keys, Streamlit secrets, or chat logs.

### 3. Run the application

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` if Streamlit does not open it automatically.

The repository contains a reviewed chunk corpus and matching Chroma vector artifact, so a knowledge-base rebuild is not required for normal use.

## Configuration

All supported settings are documented in `.env.example`.

| Variable | Default | Description |
|---|---:|---|
| `OPENAI_API_KEY` | — | Required for answer generation |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Model used for grounded synthesis |
| `ENABLE_GROUNDING_VALIDATION` | `true` | Validates generated claims against evidence |
| `MAX_EVIDENCE_CHUNKS` | `5` | Maximum chunks supplied to generation |
| `MAX_EVIDENCE_CHARS` | `6000` | Maximum evidence characters per request |
| `SESSION_REQUESTS` | `10` | Requests allowed per session window |
| `SESSION_WINDOW_SECONDS` | `60` | Session rate-limit window |
| `GENERATION_CONCURRENCY` | `2` | Concurrent generations per process |
| `DAILY_GENERATION_LIMIT` | `500` | In-process daily generation limit |

Knowledge paths can be configured using `KNOWLEDGE_INDEX_PATH`, `KNOWLEDGE_MANIFEST_PATH`, and `CHROMA_PERSIST_DIRECTORY`.

## Refreshing the knowledge base

Use this workflow when an approved page changes:

```bash
# Audit and extract the approved sources
python -m scripts.audit_sources

# Build chunks and the semantic vector index
python -m app.knowledge.indexer
python -m scripts.build_vector_index

# Approve and validate the reviewed artifact
python -m scripts.review_artifact --reviewer "your-name-or-email"
python -m scripts.validate_artifact
```

Before approval, manually review:

- the extracted content for all seven pages;
- the source titles, URLs, and page types;
- treatment and package boundaries;
- price, currency, duration, inclusion, and exclusion context; and
- candidate prices in `data/processed/price_mentions_review.jsonl`.

The corpus, manifest, and Chroma vector index form one versioned artifact and must be rebuilt, reviewed, committed, or rolled back together.

## Verification

Install the verification dependencies:

```bash
pip install -r requirements-dev.lock
```

Run the production verification commands:

```bash
# Compile application and maintenance modules
python -m compileall -q app scripts streamlit_app.py

# Validate the approved knowledge artifact
python -m scripts.validate_artifact

# Compare keyword and hybrid retrieval
python -m scripts.evaluate_retrieval --mode bm25
python -m scripts.evaluate_retrieval --mode hybrid

# Measure warm runtime performance
python -m scripts.benchmark_runtime --iterations 10

# Check installed dependency consistency and known vulnerabilities
python -m pip check
pip-audit -r requirements.lock
```

GitHub Actions runs compilation, artifact validation, dependency consistency, vulnerability auditing, and secret scanning on every push and pull request.

## Project structure

```text
app/
├── answering/       # Generation, grounding, policies, and follow-up context
├── config/          # Settings and approved source registry
├── ingestion/       # Polite allowlisted crawler
├── knowledge/       # Chunking and artifact validation
├── retrieval/       # Keyword search, Chroma, fusion, and reranking
└── runtime/         # Health checks, limits, and telemetry
data/
├── evaluation/      # Reviewed retrieval regression cases
└── processed/       # Approved chunks, manifest, and vector artifact
scripts/             # Audit, build, review, validation, and benchmark tools
streamlit_app.py     # Application entry point
```

## Safety behavior

- Answers use only retrieved content from approved PlacidWay pages.
- Factual model claims must reference valid supplied chunk IDs.
- Unsupported prices, clinics, medical facts, or package details are not invented.
- Package-specific questions cannot silently substitute evidence from another package.
- The assistant cannot diagnose, recommend a treatment, guarantee safety, or book appointments.
- Unknown information is acknowledged and the visitor is directed to the source page or PlacidWay.
- Prices and package details should always be confirmed directly with PlacidWay.

This application provides website information and is not a substitute for professional medical advice.

## Privacy

- API keys and secrets are loaded from environment variables or application secrets.
- `.env`, private crawl output, and chat logs are excluded from version control.
- Runtime telemetry records timing and safe error categories, not visitor questions or evidence text.
- No personal or medical information should be added to the knowledge corpus.

## Known limitations

- OpenAI generation time depends on external API and network latency.
- HTML extraction may retain repeated page text or miss dynamically loaded content.
- Candidate price extraction identifies review items; it does not validate quotes automatically.
- Retrieval quality depends on the source content, chunk boundaries, embedding model, and reranker.
- Live model wording can vary even when claims and citations remain constrained.
- The knowledge base must be refreshed and reviewed when approved pages change.
