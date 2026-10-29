"""Streamlit chat interface for the source-grounded PlacidWay assessment bot."""
from __future__ import annotations

import os
from pathlib import Path
import time

import streamlit as st
from dotenv import load_dotenv

from app.answering.engine import AnswerEngine
from app.answering.openai_generator import OpenAIAnswerGenerator
from app.answering.followup_context import resolve_followup_question
from app.config.settings import Settings, SettingsError
from app.retrieval.hybrid_search import HybridKnowledgeSearch
from app.retrieval.reranker import FlashRankReranker
from app.runtime.controls import DailyLimitedGenerator, SlidingWindowLimiter, trim_messages
from app.runtime.health import check_health
from app.runtime.telemetry import RuntimeEvent, emit_event

load_dotenv()
try:
    secret_values = dict(st.secrets)
except (FileNotFoundError, RuntimeError):
    secret_values = {}
try:
    settings = Settings.from_sources(os.environ, secret_values)
except SettingsError as exc:
    st.error(f"Invalid application configuration: {exc}")
    st.stop()

st.set_page_config(
    page_title="PlacidWay Knowledge Assistant",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --pw-primary: #F05A55;
        --pw-primary-dark: #C63F3B;
        --pw-surface: #ffffff;
        --pw-surface-soft: #FFF5F4;
        --pw-border: #E7D3D1;
        --pw-text: #26313D;
        --pw-muted: #5D6874;
    }
    .stApp { background: linear-gradient(180deg, #FFF8F7 0, #FFFFFF 25rem); }
    .block-container { max-width: 860px; padding-top: 2.5rem; padding-bottom: 7rem; }
    .assistant-shell {
        padding: 1.5rem 0 1.25rem;
        text-align: center;
    }
    .assistant-kicker {
        display: inline-block;
        padding: .35rem .7rem;
        border: 1px solid var(--pw-border);
        border-radius: 999px;
        color: var(--pw-primary-dark);
        background: rgba(255,255,255,.9);
        font-size: .82rem;
        font-weight: 650;
        letter-spacing: .02em;
    }
    .assistant-shell h1 {
        color: var(--pw-text);
        font-size: clamp(2rem, 5vw, 3rem);
        line-height: 1.12;
        letter-spacing: -.035em;
        margin: .85rem auto .65rem;
        max-width: 720px;
    }
    .assistant-shell p {
        color: var(--pw-muted);
        font-size: 1.05rem;
        line-height: 1.65;
        max-width: 650px;
        margin: 0 auto;
    }
    .scope-note {
        background: var(--pw-surface);
        border: 1px solid var(--pw-border);
        border-left: 4px solid var(--pw-primary);
        border-radius: 12px;
        color: var(--pw-muted);
        line-height: 1.55;
        margin: .5rem 0 1.75rem;
        padding: .9rem 1rem;
    }
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,.92);
        border: 1px solid #dce9e9;
        border-radius: 16px;
        padding: .4rem .55rem;
        box-shadow: 0 8px 28px rgba(78, 42, 40, .06);
    }
    [data-testid="stChatInput"] { border-color: var(--pw-border); }
    .stButton > button {
        min-height: 48px;
        border-radius: 12px;
        border-color: var(--pw-border);
        color: var(--pw-text);
        font-weight: 550;
        transition: border-color .15s ease, background .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover {
        border-color: var(--pw-primary);
        background: var(--pw-surface-soft);
        color: var(--pw-primary-dark);
        box-shadow: 0 5px 16px rgba(240, 90, 85, .12);
    }
    .stButton > button:focus-visible {
        outline: 3px solid rgba(240, 90, 85, .28);
        outline-offset: 2px;
    }
    @media (max-width: 640px) {
        .block-container { padding: 1.25rem 1rem 6rem; }
        .assistant-shell { padding-top: .75rem; }
        .assistant-shell h1 { font-size: 2rem; }
        .assistant-shell p { font-size: 1rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

logo_left, logo_center, logo_right = st.columns([1, 1.35, 1])
with logo_center:
    st.image("assets/placidway-logo.png", width="stretch")

st.markdown(
    """
    <section class="assistant-shell">
      <span class="assistant-kicker">Information from approved PlacidWay pages</span>
      <h1>Your guide to PlacidWay treatment information</h1>
      <p>Ask about listed packages, program durations, prices, inclusions, or the ITC clinic.</p>
    </section>
    <div class="scope-note" role="note">
      <strong>Before you continue:</strong> This assistant summarizes website information and
      does not provide medical advice, diagnoses, or bookings. Please confirm current prices and
      package details directly with PlacidWay.
    </div>
    """,
    unsafe_allow_html=True,
)

health = check_health(settings)
if not health.ready:
    st.error(health.message or "The reviewed knowledge artifact is unavailable.")
    st.stop()


@st.cache_resource
def load_engine(
    index_path: str,
    chroma_persist_directory: str,
    modified_time: float,
    api_key_present: bool,
    model: str,
    daily_generation_limit: int,
    grounding_validation_enabled: bool,
    max_evidence_chunks: int,
    max_evidence_chars: int,
) -> AnswerEngine:
    searcher = HybridKnowledgeSearch.from_jsonl(
        index_path,
        persist_directory=chroma_persist_directory,
    )
    reranker = FlashRankReranker()

    generator = None
    if api_key_present:
        try:
            base = OpenAIAnswerGenerator(
                api_key=settings.openai_api_key,
                model=model,
                grounding_validation_enabled=grounding_validation_enabled,
                max_evidence_chunks=max_evidence_chunks,
                max_evidence_chars=max_evidence_chars,
            )
            generator = DailyLimitedGenerator(
                base,
                daily_generation_limit,
            )
        except Exception as exc:
            emit_event(RuntimeEvent.failure("generator_initialization", exc))
            generator = None
    return AnswerEngine(
        searcher,
        generator=generator,
        reranker=reranker,
        max_evidence_chunks=max_evidence_chunks,
    )


api_key = settings.openai_api_key or ""
model_name = settings.openai_model

try:
    engine = load_engine(
        str(settings.index_path),
        str(settings.chroma_persist_directory),
        settings.index_path.stat().st_mtime,
        bool(api_key),
        model_name,
        settings.daily_generation_limit,
        settings.grounding_validation_enabled,
        settings.max_evidence_chunks,
        settings.max_evidence_chars,
    )
except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
    st.error(f"Could not load the knowledge index or retrieval model: {exc}")
    st.stop()

with st.sidebar:
    st.image("assets/placidway-logo.png", width=190)
    st.header("About this assistant")
    st.write("Explore information from selected PlacidWay clinic and treatment-package pages.")
    if api_key and engine.generator is not None:
        st.success("Ready to answer")
    else:
        st.warning("Answer generation is temporarily unavailable.")
    st.caption("Website information can change. Always confirm important details with PlacidWay.")
    if st.button("Start new conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.active_treatment_context = None
        st.session_state.last_assistant_answer = None
        st.rerun()
    st.markdown("[Visit the PlacidWay website](https://www.placidway.com/)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "next_message_id" not in st.session_state:
    st.session_state.next_message_id = 1
if "rate_limiter" not in st.session_state:
    st.session_state.rate_limiter = SlidingWindowLimiter(
        limit=settings.session_requests,
        window_seconds=settings.session_window_seconds,
    )

examples = [
    "What is included in the adrenal cancer outpatient package?",
    "Compare the adenocarcinoma outpatient and inpatient packages.",
    "What is the listed price and duration for the anal cancer package?",
    "What information is available about the ITC clinic?",
]
if not st.session_state.messages:
    st.markdown("#### Popular questions")
    cols = st.columns(2)
    for index, example in enumerate(examples):
        with cols[index % 2]:
            if st.button(example, key=f"example_{index}", width="stretch"):
                st.session_state.pending_question = example

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

pending = st.session_state.pop("pending_question", None)
question = st.chat_input("Ask a question about the approved pages")
if pending:
    question = pending

if question:
    message_id = st.session_state.next_message_id
    st.session_state.next_message_id += 1
    if len(question) > settings.max_question_chars:
        st.error(f"Please keep the question under {settings.max_question_chars} characters.")
        st.stop()
    allowed, retry_after = st.session_state.rate_limiter.allow(time.monotonic())
    if not allowed:
        st.error(f"Too many questions. Please retry in {max(1, round(retry_after))} seconds.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages = trim_messages(st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(question)

    # Keep only the last explicitly named treatment package as session-local context.
    # Short follow-ups reuse this topic even after several intervening follow-up turns.
    if "active_treatment_context" not in st.session_state:
        st.session_state.active_treatment_context = None
    question_for_engine, newly_named_treatment = resolve_followup_question(
        question,
        st.session_state.active_treatment_context,
        st.session_state.get("last_assistant_answer"),
    )
    if newly_named_treatment:
        st.session_state.active_treatment_context = newly_named_treatment

    with st.spinner("Searching approved pages and preparing your answer..."):
        result = engine.answer(question_for_engine, intent_question=question)
    if result.reason == "llm_generation_unavailable":
        emit_event(RuntimeEvent(
            event="answer_failed",
            response_mode=result.response_mode,
            latency_ms=round(result.latency_ms, 2),
            retrieval_ms=round(result.retrieval_ms, 2),
            generation_ms=round(result.generation_ms, 2),
            evidence_count=len(result.evidence),
            reason=result.reason,
            exception_class=result.exception_class,
        ))
        st.toast("Answer generation failed. Please retry.", icon="⚠️")
        st.stop()
    with st.chat_message("assistant"):
        st.markdown(result.text)
    st.session_state.last_assistant_answer = result.text
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.text,
        "id": message_id,
    })
    st.session_state.messages = trim_messages(st.session_state.messages)
    emit_event(RuntimeEvent(
        event="answer_completed",
        response_mode=result.response_mode,
        latency_ms=round(result.latency_ms, 2),
        retrieval_ms=round(result.retrieval_ms, 2),
        generation_ms=round(result.generation_ms, 2),
        evidence_count=len(result.evidence),
        reason=result.reason,
        exception_class=None,
    ))
