"""Answer layer with scope routing, evidence checks, optional OpenAI synthesis, and citations."""
from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Iterable

from app.answering.evidence_response import is_list_style_question
from app.retrieval.search import KnowledgeSearch, SearchResult, tokenize

QUOTE_URL = "https://www.placidway.com/"


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[str, ...]
    abstained: bool
    reason: str
    # Exact retrieved chunks accepted by the answer layer; exposed for UI inspection.
    evidence: tuple[SearchResult, ...] = ()
    latency_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    response_mode: str = "policy_response"
    exception_class: str | None = None


WEATHER_TERMS = {
    "weather", "forecast", "temperature", "rain", "snow", "humidity", "climate",
}
BOOKING_TERMS = {
    "book", "booking", "appointment", "schedule", "reserve", "reservation",
}
MEDICAL_ADVICE_TERMS = {
    "diagnose", "diagnosis", "cure", "prescribe", "prescription", "should i take",
    "what treatment should", "which treatment should", "which treatment is best",
    "what treatment is best for me", "what treatment should i choose",
    "which treatment should i choose", "what should i choose", "should i choose",
    "will i survive", "is this cancer", "medical advice",
    "recommend a treatment for me", "what should i do",
}

_GREETING_ONLY = re.compile(
    r"^(?:(?:hi+|hello|helo|hlo|hey)(?:\s+(?:there|everyone|team))?"
    r"|good\s+(?:morning|afternoon|evening)(?:\s+(?:everyone|team))?"
    r"|how\s+are\s+you(?:\s+doing)?)$",
    re.IGNORECASE,
)


def _is_standalone_greeting(text: str) -> bool:
    normalized = re.sub(r"[^\w]+", " ", text.casefold(), flags=re.UNICODE).replace("_", " ").strip()
    return bool(_GREETING_ONLY.fullmatch(normalized))


def _has_phrase(text: str, phrases: Iterable[str]) -> bool:
    lowered = text.casefold()
    return any(phrase in lowered for phrase in phrases)


def _evidence_limit(question: str) -> int:
    lowered = question.casefold()
    numbered_parts = re.findall(r"(?:^|\s)\d+[.)]\s+", question)
    if question.count("?") >= 2 or len(numbered_parts) >= 2 or any(term in lowered for term in (
        "compare", "comparison", "contrast", "difference", "differences",
        "versus", " vs ", "both", "multi-part", "multiple parts",
    )):
        return 5
    if is_list_style_question(question):
        return 4
    return 3


def _evidence_overlap(query: str, result: SearchResult) -> tuple[int, float]:
    query_terms = set(tokenize(query))
    evidence_terms = set(tokenize(f"{result.title} {result.text}"))
    if not query_terms:
        return 0, 0.0
    overlap = query_terms & evidence_terms
    return len(overlap), len(overlap) / len(query_terms)


# Aliases reflect the treatment-package entities available in the approved source set.
# Keep these aliases centralized so context matching and source selection stay consistent.
PACKAGE_ALIASES = {
    "adenocarcinoma": ("adenocarcinoma", "adeno carcinoma"),
    "adrenal": ("adrenal cancer", "adrenal"),
    "anal": ("anal cancer", "anal-cancer"),
}


def _named_package_key(text: str) -> str | None:
    """Identify the user-topic package, ignoring quoted prior assistant answers.

    Follow-up prompts may include a previous answer to resolve phrases like
    'the second one'. Package selection must not let a mistaken package name in
    that quoted answer override the user's original treatment topic.
    """
    text = re.split(r"\bthe previous assistant answer was:\s*", text, maxsplit=1, flags=re.IGNORECASE)[0]
    lowered = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    for key, aliases in PACKAGE_ALIASES.items():
        if any(re.search(rf"\b{re.escape(re.sub(r'[^a-z0-9]+', ' ', alias.casefold()).strip())}\b", lowered) for alias in aliases):
            return key
    return None


def _has_named_package_or_clinic(text: str) -> bool:
    """Return whether a price query identifies a treatment, package, or clinic."""
    lowered = text.casefold()
    if _named_package_key(text):
        return True
    return any(
        term in lowered
        for term in (
            "breast cancer", "prostate cancer", "lung cancer", "colon cancer",
            "stem cell therapy", "ozone therapy", "ivf", "fertility treatment",
            "knee replacement", "hip replacement", "dental implants", "hair transplant",
            "itc", "immunity therapy center", "clinic",
        )
    )


def _countries_named_in_comparison(question: str) -> tuple[str, ...]:
    """Find explicitly named countries in a cost-comparison question."""
    lowered = question.casefold()
    comparison_terms = ("cheaper", "compare", "comparison", "cost less", "less expensive", "price difference")
    if not any(term in lowered for term in comparison_terms):
        return ()
    countries = (
        "mexico", "turkey", "india", "thailand", "germany", "spain",
        "united states", "usa", "canada", "south korea", "japan",
        "pakistan", "malaysia", "singapore", "costa rica", "colombia",
    )
    found = []
    for country in countries:
        if re.search(rf"\b{re.escape(country)}\b", lowered):
            normalized = "united states" if country == "usa" else country
            if normalized not in found:
                found.append(normalized)
    return tuple(found) if len(found) >= 2 else ()


def _source_priority(question: str, result: SearchResult) -> int:
    """Prefer the matching package/profile source over generic comparison content."""
    q = question.casefold()
    source = result.source_id.casefold()
    package_key = _named_package_key(q)
    if package_key and package_key in source:
        return 3
    if any(term in q for term in ("clinic", "itc", "immunity therapy center")) and "clinic-profile" in source:
        return 2
    asks_explicit_comparison = any(
        term in q for term in (
            "compare", "comparison", "average", "benchmark", "price difference",
            "compared with", "versus", " vs ",
        )
    )
    if "price-comparison" in source or "comparison" in source:
        # Generic averages are weak evidence for a package-specific price, but
        # are primary evidence when the visitor explicitly asks about that page's
        # average/benchmark or compares it with another sourced figure.
        if asks_explicit_comparison:
            return 3
        if "price" in q or "cost" in q or "how much" in q:
            return -1
    return 0


class AnswerEngine:
    def __init__(
        self,
        searcher: KnowledgeSearch,
        *,
        minimum_overlap_ratio: float = 0.25,
        minimum_matched_terms: int = 1,
        generator=None,
        reranker=None,
        max_evidence_chunks: int = 5,
    ) -> None:
        if not 0 <= minimum_overlap_ratio <= 1:
            raise ValueError("minimum_overlap_ratio must be between 0 and 1")
        if minimum_matched_terms < 1:
            raise ValueError("minimum_matched_terms must be positive")
        self.searcher = searcher
        self.minimum_overlap_ratio = minimum_overlap_ratio
        self.minimum_matched_terms = minimum_matched_terms
        self.generator = generator
        self.reranker = reranker
        self.max_evidence_chunks = max_evidence_chunks

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        intent_question: str | None = None,
    ) -> Answer:
        request_started = time.perf_counter()
        # Keep visitor intent separate from the expanded retrieval query.
        query = question.strip()
        intent_query = (intent_question if intent_question is not None else question).strip()
        lower_intent = intent_query.casefold()
        if not query or not intent_query:
            return Answer(
                "Please enter a question about the PlacidWay pages covered by this assistant.",
                (), True, "empty_question"
            )

        if _is_standalone_greeting(intent_query):
            return Answer(
                "Hello! I can help you find information from the approved PlacidWay pages. What would you like to know?",
                (),
                False,
                "greeting",
            )

        lower_query = query.casefold()

        # Personal medical safety must be handled before retrieval: website text
        # cannot establish whether a treatment is safe for an individual.
        asks_personal_safety = (
            ("safe" in lower_intent or "safety" in lower_intent)
            and any(term in lower_intent for term in ("diabetes", "diabetic", "my condition", "my medical history"))
        )
        if asks_personal_safety:
            return Answer(
                "The approved PlacidWay pages cannot establish whether this treatment is safe for your individual medical situation. Please discuss the specific treatment, your diabetes, and your medical history with a qualified healthcare professional.",
                (), True, "medical_safety_not_in_sources"
            )

        # A generic inclusions question can accidentally combine details from
        # unrelated packages. Ask which package before generating an answer.
        asks_package_inclusions = any(
            phrase in lower_intent
            for phrase in ("what is included", "what's included", "what does the package include", "package inclusions")
        )
        has_package_target = _has_named_package_or_clinic(query)
        if asks_package_inclusions and not has_package_target:
            return Answer(
                "Which treatment package do you mean? I can summarize the inclusions listed on that specific approved page.",
                (),
                True,
                "clarify_package_for_inclusions",
            )

        # Clarify underspecified requests before retrieval. Searching broad
        # terms for these questions tends to surface unrelated packages, which
        # makes the inspection panel look as if that content supports the reply.
        price_claim = re.search(r"\$\s*\d[\d,]*(?:\.\d{1,2})?", intent_query)
        if price_claim:
            without_claim = intent_query[:price_claim.start()] + intent_query[price_claim.end():]
            if not _has_named_package_or_clinic(without_claim):
                return Answer(
                    "I can't verify that amount without knowing which treatment or package you mean. Which PlacidWay treatment or package are you asking about?",
                    (), True, "ambiguous_price_target",
                )

        asks_total_cost = any(
            phrase in lower_intent
            for phrase in (
                "complete treatment program", "complete treatment cost",
                "total treatment cost", "total cost", "overall cost",
                "how much money should i expect to pay",
                "how much does the treatment cost",
                "how much is the treatment",
            )
        )
        if asks_total_cost and not _has_named_package_or_clinic(query):
            return Answer(
                "Which treatment or package are you asking about? Costs differ by package, and I can summarize the price listed on the matching approved page.",
                (), True, "clarify_package_for_total_cost",
            )

        compared_countries = _countries_named_in_comparison(intent_query)
        if compared_countries:
            named_treatment = any(
                term in lower_intent
                for term in (
                    "adenocarcinoma", "adrenal", "anal cancer", "cancer treatment",
                    "stem cell", "immunotherapy", "ozone therapy", "dental",
                    "cosmetic", "ivf", "fertility", "knee replacement",
                    "hip replacement", "treatment package",
                )
            )
            if not named_treatment:
                return Answer(
                    "I can't determine which country is cheaper without comparable prices for the same treatment. Which treatment or package are you comparing? The approved pages may not provide pricing for both countries.",
                    (), True, "insufficient_comparable_country_pricing",
                )

        # Retrieve only after checking whether the request is specific enough
        # to identify the relevant source material.
        retrieval_query = query
        price_claim = re.search(r"\$\s*\d[\d,]*(?:\.\d{1,2})?", intent_query)
        if price_claim:
            # Remove the visitor's amount, not amounts quoted in prior context.
            retrieval_query = query.replace(price_claim.group(0), "", 1)

        # The corpus is small; a compact deterministic pool is enough and avoids
        # repeated list-query searches and reranking work.
        candidate_k = max(top_k * 6, 30)
        retrieval_started = time.perf_counter()
        if any(phrase in lower_intent for phrase in ("include", "included", "inclusions")):
            retrieval_query += " included inclusions lab imaging therapies nutrition medications follow-up"
        results = self.searcher.search(retrieval_query, top_k=candidate_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        # If a short comparison query fails to retrieve evidence, broaden the
        # search with the topic and comparison concepts, not a canned answer.
        if not results and any(term in lower_query for term in ("cheaper", "compare", "comparison", "cost less", "less expensive")):
            results = self.searcher.search(
                f"{retrieval_query} treatment price cost Mexico Turkey",
                top_k=candidate_k,
            )

        ranked = sorted(
            results,
            key=lambda result: (_source_priority(query, result), result.score),
            reverse=True,
        )

        # Country comparisons require evidence about both named countries and
        # pricing. Never infer a winner from unrelated prices.
        if compared_countries:
            evidence_text = " ".join(
                f"{item.title} {item.source_id} {item.text}" for item in ranked
            ).casefold()
            countries_present = all(country in evidence_text for country in compared_countries)
            has_price_context = any(
                term in evidence_text
                for term in ("price", "cost", "usd", "$", "pricing", "package")
            )
            if not named_treatment or not countries_present or not has_price_context:
                return Answer(
                    "I can't determine which country is cheaper without comparable prices for the same treatment. Which treatment or package are you comparing? The approved pages may not provide pricing for both countries.",
                    (),
                    True,
                    "insufficient_comparable_country_pricing",
                    tuple(ranked[:5]),
                )

        # Named-package questions should prefer evidence from that package.
        package_source = _named_package_key(query)
        if package_source:
            exact_package_results = [
                result for result in ranked
                if package_source in result.source_id.casefold()
                and result.page_type == "treatment_package"
            ]
            if exact_package_results:
                ranked = exact_package_results
            else:
                return Answer(
                    "I couldn't find enough information from the approved page for that specific treatment package. I don't want to substitute details from a different package. Please check the package page or contact PlacidWay for clarification.",
                    (),
                    True,
                    "named_package_evidence_missing",
                    tuple(ranked[:5]),
                )

        accepted: list[SearchResult] = []
        is_contextual_followup = lower_query.startswith("follow-up about the same topic")
        for result in ranked:
            matched_terms, ratio = _evidence_overlap(retrieval_query, result)
            # Contextual follow-ups contain the previous topic plus the short
            # follow-up, so their overall overlap ratio is naturally diluted.
            # Keep the topic match requirement, but don't apply the same ratio
            # threshold to these expanded queries.
            relevant_enough = (
                matched_terms >= self.minimum_matched_terms
                and (ratio >= self.minimum_overlap_ratio or is_contextual_followup)
            ) or "vector" in result.retrieval_sources
            if relevant_enough:
                accepted.append(result)

        target_chunk_count = 6
        if package_source and len(accepted) < target_chunk_count:
            accepted_ids = {item.chunk_id for item in accepted}
            for result in ranked:
                if (
                    result.chunk_id not in accepted_ids
                    and package_source in result.source_id.casefold()
                    and result.page_type == "treatment_package"
                ):
                    accepted.append(result)
                    accepted_ids.add(result.chunk_id)
                    if len(accepted) >= target_chunk_count:
                        break

        evidence_limit = min(_evidence_limit(intent_query), self.max_evidence_chunks)
        if self.reranker is not None:
            rerank_started = time.perf_counter()
            accepted = self.reranker.rerank(query, accepted, top_n=evidence_limit)
            retrieval_ms += (time.perf_counter() - rerank_started) * 1000
        else:
            accepted = accepted[:evidence_limit]

        if _has_phrase(intent_query, MEDICAL_ADVICE_TERMS):
            return Answer(
                "I can share information published on the listed PlacidWay pages, but I can't diagnose a condition or recommend a treatment for your personal medical situation. Please consult a qualified healthcare professional. For clinic or package information, I can help summarize the source pages.",
                (), True, "medical_advice"
            )

        meaningful_tokens = set(tokenize(intent_query))
        if meaningful_tokens & WEATHER_TERMS:
            return Answer(
                "I can only answer questions supported by the approved PlacidWay pages, so I can't provide weather or forecast information. I can help with the clinic profile, listed treatment packages, and pricing information on those pages.",
                (), True, "out_of_scope_weather"
            )

        if meaningful_tokens & BOOKING_TERMS:
            return Answer(
                "I can't make or confirm appointments or bookings. You can contact PlacidWay through its official website to ask about availability or request a free quote: " + QUOTE_URL,
                (QUOTE_URL,), True, "unsupported_booking"
            )

        if not accepted:
            return Answer(
                "I couldn't find enough supporting information in the approved PlacidWay pages to answer that confidently. Please check the source pages or contact PlacidWay through its official website for a free quote or clarification: " + QUOTE_URL,
                (QUOTE_URL,), True, "insufficient_evidence"
            )

        unique_urls: list[str] = []
        for result in accepted:
            if result.source_url not in unique_urls:
                unique_urls.append(result.source_url)

        if self.generator is not None:
            generation_started = time.perf_counter()
            try:
                generated = self.generator.generate(
                    query, accepted, package_key=package_source
                )
                reason = {
                    "answer": "grounded_llm_answer",
                    "greeting": "llm_greeting",
                    "out_of_scope": "llm_out_of_scope",
                    "insufficient_evidence": "llm_insufficient_evidence",
                }[generated.response_type]
            except Exception as exc:
                generation_ms = (time.perf_counter() - generation_started) * 1000
                return Answer(
                    "AI answer generation is temporarily unavailable. Please try again.",
                    tuple(unique_urls), True, "llm_generation_unavailable", tuple(accepted),
                    latency_ms=(time.perf_counter() - request_started) * 1000,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    response_mode="llm_generation_unavailable",
                    exception_class=exc.__class__.__name__,
                )
            generation_ms = (time.perf_counter() - generation_started) * 1000
        else:
            return Answer(
                "AI answer generation is temporarily unavailable. Please try again.",
                tuple(unique_urls), True, "llm_generation_unavailable", tuple(accepted),
                latency_ms=(time.perf_counter() - request_started) * 1000,
                retrieval_ms=retrieval_ms,
                response_mode="llm_generation_unavailable",
                exception_class="GeneratorNotConfiguredError",
            )

        if generated.response_type != "answer":
            return Answer(
                generated.text,
                (),
                generated.response_type != "greeting",
                reason,
                tuple(accepted),
                latency_ms=(time.perf_counter() - request_started) * 1000,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                response_mode=reason,
            )

        cited_ids = set(generated.cited_chunk_ids)
        cited_results = [item for item in accepted if item.chunk_id in cited_ids]
        cited_urls: list[str] = []
        cited_source_lines: list[str] = []
        for result in cited_results:
            if result.source_url not in cited_urls:
                cited_urls.append(result.source_url)
                cited_source_lines.append(f"- [{result.title}]({result.source_url})")
        if not cited_source_lines:
            for result in accepted:
                if result.source_url not in cited_urls:
                    cited_urls.append(result.source_url)
                    cited_source_lines.append(f"- [{result.title}]({result.source_url})")

        response = generated.text + "\n\n**Sources**\n" + "\n".join(cited_source_lines)
        return Answer(
            response,
            tuple(cited_urls),
            False,
            reason,
            tuple(accepted),
            latency_ms=(time.perf_counter() - request_started) * 1000,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms if self.generator is not None else 0.0,
            response_mode=reason,
        )

    @staticmethod
    def _excerpt_fallback(result: SearchResult) -> str:
        excerpt = re.sub(r"\s+", " ", result.text).strip()
        if len(excerpt) > 650:
            excerpt = excerpt[:647].rsplit(" ", 1)[0] + "..."
        return "Here is the relevant information I found on the approved PlacidWay pages:\n\n" + excerpt
