"""Deterministic evidence selection and safe extractive degradation."""
from __future__ import annotations

import re
from typing import Sequence

from app.retrieval.search import SearchResult, tokenize

_PROMOTIONAL = re.compile(
    r"\b(cure[sd]?|guarantee[sd]?|proven effective|safe for everyone|no side effects|"
    r"miracle|success rate|eliminates? cancer)\b",
    re.IGNORECASE,
)
_ATTRIBUTE_TERMS = {
    "duration": {"duration", "long", "length", "weeks", "days", "months"},
    "price": {"price", "cost", "usd", "dollar", "dollars", "expensive", "much"},
    "success": {"success", "rate", "effective", "effectiveness", "outcome", "outcomes"},
    "inclusions": {"include", "included", "includes", "inclusions"},
    "location": {"where", "location", "located", "address"},
}


def is_list_style_question(question: str) -> bool:
    lowered = re.sub(r"\s+", " ", question.casefold()).strip()
    current = lowered.rsplit("the user now asks", 1)[-1]
    if re.match(r"^(?:is|are|does|do|can|has|have)\b", current):
        return False
    return bool(
        re.search(r"\b(?:list|enumerate)\b", current)
        or re.search(r"\bwhat are the\b", current)
        or re.search(r"\bwhich (?:procedures|services|treatments|items)\b", current)
        or re.search(r"\bwhat(?: is|'s) included\b", current)
        or re.search(r"\bwhat does .+ include\b", current)
        or "package inclusions" in current
    )


def _attributes_in_text(text: str) -> set[str]:
    terms = set(tokenize(text))
    found = {name for name, values in _ATTRIBUTE_TERMS.items() if terms & values}
    if re.search(r"(?:\$\s*\d|\b\d[\d,]*\s*(?:USD|US dollars)\b)", text, re.I):
        found.add("price")
    if re.search(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:days?|weeks?|months?)\b", text, re.I):
        found.add("duration")
    return found


def _package_table_excerpt(question: str, text: str) -> str | None:
    """Extract one complete program row from newline-separated package tables."""
    normalized_question = question.casefold().replace("-", "")
    target = next(
        (name for name in ("outpatient", "inpatient") if name in normalized_question),
        None,
    )
    if target is None:
        return None
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        normalized_line = line.casefold().replace("-", "")
        if target not in normalized_line or "program" not in normalized_line:
            continue
        row = [line]
        for following in lines[index + 1:index + 5]:
            normalized_following = following.casefold().replace("-", "")
            if "program" in normalized_following and any(
                name in normalized_following for name in ("outpatient", "inpatient")
            ):
                break
            row.append(following)
        combined = " | ".join(row)
        has_duration = bool(re.search(r"\b\d+\s*(?:days?|weeks?|months?)\b", combined, re.I))
        has_price = bool(re.search(r"(?:\$\s*\d|\b\d[\d,]*\s*USD\b)", combined, re.I))
        if has_duration and has_price:
            return combined
    return None


def safe_text_prefix(text: str, limit: int) -> str:
    """Return only complete sentences; never manufacture facts by character slicing."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    kept: list[str] = []
    used = 0
    for sentence in sentences:
        sentence = sentence.strip()
        added = len(sentence) + (1 if kept else 0)
        if not sentence or used + added > limit:
            break
        kept.append(sentence)
        used += added
    if kept:
        return " ".join(kept)
    # Some extracted headings/lists contain no sentence punctuation. They may
    # be compacted at a word boundary, but never truncate a punctuated sentence.
    if not re.search(r"[.!?]", text):
        boundary = text.rfind(" ", 0, limit + 1)
        return text[:boundary].strip() if boundary > 0 else ""
    return ""


def select_evidence(
    question: str,
    candidates: Sequence[SearchResult],
    package_key: str | None,
    max_chunks: int = 3,
    max_chars: int = 6000,
) -> tuple[SearchResult, ...]:
    eligible = [
        item for item in candidates
        if package_key is None or (
            package_key in item.source_id.casefold()
            and item.page_type == "treatment_package"
        )
    ]
    eligible.sort(key=lambda item: (-item.score, item.chunk_id))
    query_terms = {
        term for term in tokenize(question)
        if len(term) > 2 and term not in {"what", "which", "this", "that", "the", "and", "for"}
    }
    coverable_terms = query_terms & {
        term for item in eligible for term in tokenize(item.text)
    }
    requested_attributes = _attributes_in_text(question)
    list_style = is_list_style_question(question)
    selected: list[SearchResult] = []
    covered_terms: set[str] = set()
    covered_attributes: set[str] = set()
    remaining = max_chars
    for item in eligible:
        if len(selected) >= max_chunks or remaining <= 0:
            break
        item_terms = set(tokenize(item.text))
        item_attributes = _attributes_in_text(item.text)
        adds_coverage = (
            ((item_terms & coverable_terms) - covered_terms)
            or ((item_attributes & requested_attributes) - covered_attributes)
        )
        if selected and not list_style and not adds_coverage:
            continue
        text = safe_text_prefix(item.text, remaining)
        if not text:
            continue
        selected.append(SearchResult(
            item.chunk_id, item.source_id, item.page_type,
            item.source_url, item.title, text, item.score,
        ))
        covered_terms.update(item_terms & coverable_terms)
        covered_attributes.update(item_attributes & requested_attributes)
        remaining -= len(text)
        if (
            not list_style
            and covered_terms >= coverable_terms
            and covered_attributes >= requested_attributes
        ):
            break
    return tuple(selected)


def build_evidence_response(question: str, evidence: Sequence[SearchResult]) -> str | None:
    query_terms = set(tokenize(question))
    if not query_terms:
        return None
    excerpts: list[str] = []
    seen_excerpt_text: set[str] = set()
    requested_attributes = {
        name for name, terms in _ATTRIBUTE_TERMS.items() if query_terms & terms
    }
    for item in evidence:
        table_excerpt = _package_table_excerpt(question, item.text)
        if table_excerpt and table_excerpt not in seen_excerpt_text:
            seen_excerpt_text.add(table_excerpt)
            excerpts.append(f"- {table_excerpt} ([{item.title}]({item.source_url}))")
            if len(excerpts) == 3:
                break
            continue
        sentences = re.split(r"(?<=[.!?])\s+|\n+", item.text)
        for sentence in sentences:
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if not sentence or _PROMOTIONAL.search(sentence):
                continue
            sentence_terms = set(tokenize(sentence))
            answers_attribute = all(
                sentence_terms & _ATTRIBUTE_TERMS[name] for name in requested_attributes
            )
            if query_terms & sentence_terms and answers_attribute:
                excerpts.append(f"- {sentence} ([{item.title}]({item.source_url}))")
                break
        if len(excerpts) == 3:
            break
    if not excerpts:
        return None
    return "Here is the directly retrieved information from the approved pages:\n\n" + "\n".join(excerpts)
