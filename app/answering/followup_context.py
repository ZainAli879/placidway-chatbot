"""Generic session-local context resolution for short chatbot follow-ups."""
from __future__ import annotations

import re


_FOLLOWUP_STARTS = (
    "and ", "what about", "how about", "how long", "how much",
    "does it", "is it", "are they", "are those", "do those",
    "what does that", "what are the", "can you explain",
    "can you also", "can u also", "could you also", "please also",
    "tell me more", "why is that", "when is it", "where is it",
)
_FOLLOWUP_REFERENCES = (
    "it", "they", "them", "that", "those", "these", "there", "same",
    "also", "more", "cost", "price", "recovery", "duration", "included",
)
_PACKAGE_PATTERNS = {
    "adenocarcinoma": r"\badenocarcinoma\b",
    "adrenal": r"\badrenal(?: cancer)?\b",
    "anal": r"\banal cancer\b",
}


def _explicit_package(text: str) -> str | None:
    lowered = text.casefold()
    return next((key for key, pattern in _PACKAGE_PATTERNS.items() if re.search(pattern, lowered)), None)


def _looks_like_followup(question: str) -> bool:
    text = (question or "").strip().casefold()
    words = text.split()
    if not words:
        return False

    # Explicit follow-up openers take precedence over length: users often ask
    # natural, longer follow-ups such as "Can you also provide what's included
    # in the outpatient program you just mentioned?"
    if text.startswith(_FOLLOWUP_STARTS):
        return True

    if len(words) > 12:
        return False

    # Broad openers such as "what is the..." are not sufficient on their own:
    # "What is the weather in Cancun?" is a complete new question.

    # Short questions with explicit anaphora or follow-up terms can use context.
    return len(words) <= 7 and any(
        re.search(rf"\b{re.escape(reference)}\b", text)
        for reference in _FOLLOWUP_REFERENCES
    )


def _followup_intent(question: str) -> str:
    text = question.casefold()
    if any(term in text for term in ("cost", "price", "expensive", "how much")):
        return "price or cost"
    if any(term in text for term in ("recovery", "recover", "healing time")):
        return "recovery time"
    if any(term in text for term in ("include", "included", "package")):
        return "package inclusions"
    if any(term in text for term in ("duration", "how long", "length")):
        return "duration"
    if any(term in text for term in ("clinic", "treatments", "services", "offer")):
        return "clinic information"
    return "the requested detail"


def resolve_followup_question(
    question: str,
    active_context: str | None,
    previous_answer: str | None = None,
) -> tuple[str, str | None]:
    """Resolve follow-ups against the latest substantive user topic.

    The context is the user's actual prior question, not a hardcoded entity list.
    Returns (retrieval question, context to store). A new standalone question
    replaces prior context; a follow-up retains it. Memory stays in the caller's
    session state and should be cleared with the conversation.
    """
    current = (question or "").strip()
    if not current:
        return current, None

    current_package = _explicit_package(current)
    context_package = _explicit_package(active_context or "")
    if current_package and current_package != context_package:
        return current, current

    if active_context and _looks_like_followup(current):
        intent = _followup_intent(current)
        answer_context = ""
        if previous_answer and re.search(
            r"\b(second|third|fourth|first|former|latter|one|ones|that item|that option)\b",
            current.casefold(),
        ):
            # Ordinal/deictic references should resolve against the prior answer,
            # not be guessed as the second program or package.
            answer_context = (
                f" The previous assistant answer was: {previous_answer}. "
                "Resolve references such as 'the second one' against that answer's "
                "list or options. If the reference is not unambiguous, ask which "
                "item the visitor means instead of guessing."
            )
        resolved = (
            f"Follow-up about the same topic as this previous question: "
            f"{active_context}.{answer_context} The user now asks for {intent}: {current}. "
            "Answer only from relevant approved-page evidence; preserve the same "
            "named treatment/package and do not substitute another package or "
            "reinterpret a list item as a program option. Do not infer missing details."
        )
        return resolved, None

    # A standalone question becomes the new context, even when out of scope.
    return current, current
