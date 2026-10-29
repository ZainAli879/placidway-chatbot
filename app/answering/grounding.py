"""Validate model claims against the exact evidence supplied to generation."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Sequence

from app.retrieval.search import SearchResult

_LIMITATION_PATTERNS = (
    r"i (?:could not|couldn't|can't) verify [^.]+ from the approved (?:placidway )?pages\.?",
    r"the (?:approved pages|supplied evidence) (?:do|does) not provide (?:that|this|the requested) detail\.?",
    r"please confirm (?:prices?|package details) (?:directly )?with placidway\.?",
    r"this is website information, not medical advice\.?",
)
_UNSAFE_MEDICAL = re.compile(
    r"\b(cure[sd]?|guarantee[sd]?|completely safe|safe for everyone|"
    r"no side effects|eliminates? cancer|will remove (?:your |the )?tumou?r)\b",
    re.I,
)
_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million",
}
_NUMBER_WORD_VALUES = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "million": "1000000",
}
_MONTH_VALUES = {
    "january": "1", "february": "2", "march": "3", "april": "4",
    "may": "5", "june": "6", "july": "7", "august": "8",
    "september": "9", "october": "10", "november": "11", "december": "12",
}
def _amounts(value: str) -> set[str]:
    patterns = (
        r"\$\s*(\d[\d,]*(?:\.\d{1,2})?)",
        r"\b(\d[\d,]*(?:\.\d{1,2})?)\s*(?:USD|US dollars)\b",
        r"\bUSD\s*(\d[\d,]*(?:\.\d{1,2})?)",
    )
    return {
        _normalized_decimal(match)
        for pattern in patterns
        for match in re.findall(pattern, value, re.IGNORECASE)
    }


def _normalized_decimal(raw: str) -> str:
    try:
        return format(Decimal(raw.replace(",", "")).normalize(), "f")
    except InvalidOperation:
        return raw.replace(",", "")


def _number_word_values(value: str) -> set[str]:
    words = re.findall(r"[a-z]+", value.casefold())
    values: set[str] = set()
    index = 0
    while index < len(words):
        if words[index] not in _NUMBER_WORDS:
            index += 1
            continue
        phrase: list[str] = []
        while index < len(words) and words[index] in _NUMBER_WORDS:
            phrase.append(words[index])
            index += 1
        total = 0
        current = 0
        for word in phrase:
            number = int(_NUMBER_WORD_VALUES[word])
            if word == "hundred":
                current = max(1, current) * 100
            elif word in {"thousand", "million"}:
                scale = number
                total += max(1, current) * scale
                current = 0
            else:
                current += number
        values.add(str(total + current))
    return values


def _month_values(value: str) -> set[str]:
    lowered = value.casefold()
    values = {
        numeric
        for month, numeric in _MONTH_VALUES.items()
        if month != "may" and re.search(rf"\b{month}\b", lowered)
    }
    if re.search(r"\bmay\s+\d{4}\b", lowered):
        values.add(_MONTH_VALUES["may"])
    return values


def _has_unsafe_medical_claim(value: str) -> bool:
    for match in _UNSAFE_MEDICAL.finditer(value):
        prefix = value[max(0, match.start() - 24):match.start()]
        if re.search(
            r"(?:\b(?:does|do|did|is|are|was|were|will|can|could|should|would)\s+not\s+"
            r"|\b(?:cannot|can't|never|not)\s+)$",
            prefix,
            re.I,
        ):
            continue
        return True
    return False


def _supported(claim: str, evidence_text: str) -> bool:
    def numeric_values(value: str) -> set[str]:
        found = set()
        for match in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", value):
            found.add(_normalized_decimal(match))
        return found

    claim_values = (
        _number_word_values(claim)
        | _month_values(claim)
        | numeric_values(claim)
    )
    evidence_values = (
        _number_word_values(evidence_text)
        | _month_values(evidence_text)
        | numeric_values(evidence_text)
    )
    return (
        claim_values.issubset(evidence_values)
        and not _has_unsafe_medical_claim(claim)
    )


def _allowed_limitation(text: str) -> bool:
    return any(re.fullmatch(pattern, text.strip(), re.I) for pattern in _LIMITATION_PATTERNS)


def validate_generation(payload: object, evidence: Sequence[SearchResult], package_key: str | None) -> str | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("claims"), list):
        return None
    by_id = {item.chunk_id: item for item in evidence}
    rendered: list[str] = []
    for claim in payload["claims"]:
        if not isinstance(claim, dict):
            return None
        text = str(claim.get("text", "")).strip()
        chunk_ids = claim.get("chunk_ids")
        if not text or not isinstance(chunk_ids, list) or any(not isinstance(value, str) for value in chunk_ids):
            return None
        if not chunk_ids:
            if not _allowed_limitation(text):
                return None
            rendered.append(text)
            continue
        cited = [by_id.get(chunk_id) for chunk_id in chunk_ids]
        if any(item is None for item in cited):
            return None
        cited_results = [item for item in cited if item is not None]
        if package_key and any(package_key not in item.source_id.casefold() for item in cited_results):
            return None
        evidence_text = " ".join(item.text for item in cited_results)
        if not _amounts(text).issubset(_amounts(evidence_text)) or not _supported(text, evidence_text):
            return None
        links: list[str] = []
        for item in cited_results:
            link = f"[{item.title}]({item.source_url})"
            if link not in links:
                links.append(link)
        rendered.append(f"{text} ({'; '.join(links)})")
    return "\n\n".join(rendered) if rendered else None
