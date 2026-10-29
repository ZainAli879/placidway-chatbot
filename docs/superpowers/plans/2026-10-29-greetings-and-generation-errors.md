# Greetings and Generation Error Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Handle standalone greetings professionally and expose privacy-safe generation failure categories in telemetry.

**Architecture:** Route only greeting-only input to an immediate policy response before retrieval. Represent generator validation failures with typed exceptions, preserve a safe category on `Answer`, and send that category to existing JSON telemetry.

**Tech Stack:** Python, pytest, Streamlit, OpenAI Responses API

**Spec:** `docs/superpowers/specs/2026-10-29-greetings-and-generation-errors-design.md`

## Global Constraints

- Do not change booking, medical, weather, unrelated-query, retrieval, reranking, or evidence-selection behavior.
- Do not add deterministic factual answer fallback behavior.
- Never log prompts, evidence, API keys, exception messages, or sensitive configuration.
- A greeting combined with a substantive question must use the grounded pipeline.

## Review Focus

- Greeting punctuation and capitalization should still produce the greeting response.
- `Hi, what is included...` must not be intercepted as greeting-only.
- Provider subclasses must be recorded by class name without their message.
- Empty output, invalid JSON, and grounding rejection must produce different categories.
- Successful responses must keep `exception_class` empty.

---

### Task 1: Standalone greeting routing

**Files:**
- Modify: `app/answering/engine.py`
- Test: `tests/test_answer_engine.py`

**Interfaces:**
- Produces: greeting-only `Answer` with `reason="greeting"`, no evidence, and `response_mode="policy_response"`.

- [ ] Add failing tests for `Hi!`, `Good morning`, and `Hi, what is included in the adrenal cancer package?`.
- [ ] Run the focused greeting tests and verify the greeting cases fail while the combined question reaches the existing pipeline.
- [ ] Add a narrow greeting-only classifier and early route in `AnswerEngine.answer`.
- [ ] Run `tests/test_answer_engine.py` and verify all cases pass.

### Task 2: Generation failure categorization and telemetry

**Files:**
- Modify: `app/answering/openai_generator.py`
- Modify: `app/answering/engine.py`
- Modify: `streamlit_app.py`
- Test: `tests/test_openai_generator.py`
- Test: `tests/test_answer_engine.py`

**Interfaces:**
- Produces: typed generator errors with stable safe categories; `Answer.exception_class: str | None`; telemetry populated from that field.

- [ ] Add failing tests for empty output, invalid structured output, grounding rejection, and provider exception preservation.
- [ ] Run focused tests and verify failures reflect missing categories.
- [ ] Add typed generator exceptions, map caught errors into `Answer.exception_class`, and use it in failure telemetry.
- [ ] Run focused tests and verify all new cases pass.
- [ ] Run the full test suite, compilation, dependency check, and `git diff --check`.
- [ ] Commit the scoped changes without including the user-owned index manifest change.

