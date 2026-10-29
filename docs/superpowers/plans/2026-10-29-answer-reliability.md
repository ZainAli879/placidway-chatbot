# Answer Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve structured-answer reliability and grounding without reducing retrieval recall.

**Architecture:** Provider response-state handling belongs in the OpenAI generator, evidence-policy validation stays in the grounding module, and runtime limits flow from Settings through Streamlit into the engine/generator. FlashRank remains the final relevance judge and exposes its score without adding an uncalibrated cutoff.

**Tech Stack:** Python, OpenAI Responses API, Chroma, FlashRank, Streamlit, pytest

**Spec:** `docs/superpowers/specs/2026-10-29-answer-reliability-design.md`

## Global Constraints

- Do not add a generation timeout or extractive answer fallback.
- Do not reduce existing retrieval recall or change policy-response scope.
- Preserve the user-owned runtime Chroma modifications.

## Review Focus

- Incomplete response retries exactly once and complete responses do not retry.
- Grounded paraphrases pass while invented numbers and citations fail.
- Package identity reaches validation through `ControlledGenerator`.
- Settings actually control chunk/character budgets.
- FlashRank score preservation does not alter result ordering.

---

### Task 1: Provider-aware structured output

**Files:** `app/answering/openai_generator.py`, `tests/test_openai_generator.py`

- [ ] Add failing completion-state and one-retry tests.
- [ ] Implement response-state inspection and a single larger-budget recovery request.
- [ ] Run focused tests.

### Task 2: Practical grounding and package propagation

**Files:** `app/answering/grounding.py`, `app/answering/engine.py`, `app/runtime/controls.py`, `app/config/settings.py`, `streamlit_app.py`, related tests

- [ ] Add failing paraphrase, numeric, package-propagation, and default-setting tests.
- [ ] Remove lexical rejection, enable exact grounding by default, and propagate package identity.
- [ ] Wire configured evidence limits.
- [ ] Run focused tests.

### Task 3: Reranker score preservation and verification

**Files:** `app/retrieval/reranker.py`, `tests/test_reranker.py`

- [ ] Add a failing test for returned FlashRank scores and stable ordering.
- [ ] Preserve scores in returned `SearchResult` objects.
- [ ] Run retrieval evaluation and the full suite, review the diff, and commit without Chroma runtime files.
