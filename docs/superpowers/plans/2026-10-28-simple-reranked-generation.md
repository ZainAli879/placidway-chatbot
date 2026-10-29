# Simple Reranked Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace deterministic answer recovery with a simple retrieve, rerank, and AI-generate path.

**Architecture:** BM25 retrieves a candidate pool. A lightweight intent-aware scorer reranks same-package chunks and sends the top three focused or top four list chunks to the existing grounded generator. Provider or validation errors return an explicit unavailable response; they never synthesize a different deterministic answer.

**Tech Stack:** Python, OpenAI Responses API, pytest.

**Spec:** User instruction on October 28, 2026.

## Global Constraints

- AI receives only retrieved chunks from the matching package.
- Use three chunks for focused questions and four for list questions.
- Use the lightweight local TinyBERT cross-encoder and its required model-file cache; do not add an answer cache, timeout, or deterministic answer path.
- Keep claim-level grounding validation.

## Review Focus

- Adrenal outpatient-inclusions query ranks the inclusion chunk first.
- Price/duration query still ranks the pricing chunk first.
- Cross-package chunks never enter the AI prompt.
- AI failure never returns an unrelated extracted answer.
- Focused/list caps are exactly three/four.

### Task 1: Reranked AI-only answer path

- [x] Add failing rerank and provider-failure tests.
- [x] Implement passage-level semantic reranking and exact three/four limits.
- [x] Remove deterministic recovery from the runtime answer path.
- [x] Run the complete suite and benchmark.
- [x] Commit and review.
