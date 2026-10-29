# Answer Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve answer readability through model instructions without code-based rewriting or loss of supported detail.

**Architecture:** Update only the OpenAI generation instruction. Preserve the JSON schema, grounding renderer, retrieval pipeline, evidence budgets, and UI rendering.

**Tech Stack:** Python, OpenAI Responses API, pytest

**Spec:** `docs/superpowers/specs/2026-10-29-answer-presentation-design.md`

## Global Constraints

- Prompt-only presentation change.
- Do not omit materially relevant supported details for brevity.
- Do not change retrieval, reranking, grounding, fallbacks, or token budgets.

## Review Focus

- Multi-item inclusions should request Markdown bullets.
- Simple factual answers should remain concise paragraphs.
- Prices, durations, qualifiers, and exceptions must remain protected.
- Structured JSON must remain unchanged.
- Prompt must not request duplicate citations or source titles.

---

### Task 1: Generation presentation guidance

**Files:**
- Modify: `app/answering/openai_generator.py`
- Test: `tests/test_openai_generator.py`

**Interfaces:**
- Consumes: existing `OpenAIAnswerGenerator.generate(...)` request construction
- Produces: the same strict `claims` JSON response schema

- [ ] Add a failing test asserting the prompt requests conditional Markdown bullets and protects all supported details.
- [ ] Run the targeted test and confirm it fails because the guidance is absent.
- [ ] Update only the generation instruction text.
- [ ] Run the targeted test and full suite.
- [ ] Commit and publish the verified change.

### Task 2: Professional source presentation

**Files:**
- Modify: `app/answering/grounding.py`
- Modify: `app/answering/openai_generator.py`
- Modify: `app/answering/engine.py`
- Test: `tests/test_grounding.py`
- Test: `tests/test_openai_generator.py`
- Test: `tests/test_answer_engine.py`

**Interfaces:**
- Consumes: validated claim text and trusted `SearchResult` source metadata
- Produces: clean answer body followed by one deduplicated Markdown source list

- [ ] Add failing tests for clean claim text, titled source links, deduplication, and removal of the repeated disclaimer.
- [ ] Remove inline links while preserving chunk-ID validation.
- [ ] Render trusted source titles as Markdown links once in the engine.
- [ ] Run targeted tests and the complete suite.
- [ ] Commit and publish the verified change.

