# Generation Latency Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce generation latency without reducing grounded answer coverage.

**Architecture:** Select between one and five ranked evidence chunks according to query breadth and marginal lexical coverage, with an absolute 6,000-character ceiling. Use `gpt-6-luna` by default and cap ordinary structured output at 180 tokens while allowing list-style questions 250 tokens.

**Tech Stack:** Python, OpenAI Responses API, pytest.

**Spec:** User-approved scope in the October 28, 2026 conversation; no separate design file because the user explicitly requested immediate implementation of only changes 2, 3, and 4.

## Global Constraints

- Preserve retrieved-chunk grounding and claim validation.
- Use one to five chunks, never more than 6,000 evidence characters.
- Do not add streaming, caching, deterministic routing, or timeouts.
- Default model is `gpt-6-luna`; environment configuration may override it.
- Ordinary output budget is 180 tokens; genuine list-style questions may use 250.

## Review Focus

- A narrow price/duration question should not send irrelevant chunks.
- A multi-part/list question must retain enough chunks for complete coverage.
- Dynamic selection must never exceed five chunks or 6,000 characters.
- Model overrides must continue to work.
- Output-budget classification must not treat every question containing “included” as an unlimited list.

---

### Task 1: Adaptive evidence and generation budget

**Files:**
- Modify: `app/answering/evidence_response.py`
- Modify: `app/answering/engine.py`
- Modify: `app/answering/openai_generator.py`
- Modify: `app/config/settings.py`
- Modify: `.env.example`
- Modify: `README.md`
- Test: `tests/test_evidence_response.py`
- Test: `tests/test_list_retrieval.py`
- Test: `tests/test_openai_generator.py`
- Test: `tests/test_settings.py`

**Interfaces:**
- Produces: adaptive selection returning `tuple[SearchResult, ...]` of length 1–5.
- Produces: `_output_token_budget(question: str) -> int` returning 180 or 250.

- [x] Write failing tests for narrow, multi-part, five-chunk, model-default, override, and token-budget behavior.
- [x] Run focused tests and confirm failures represent the missing behavior.
- [x] Implement marginal-coverage selection and propagate the five-chunk cap.
- [x] Change the default model and adaptive output budget.
- [x] Update deployment documentation and examples.
- [x] Run focused tests and the complete suite.
- [x] Commit the implementation.
