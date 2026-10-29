# Answer Reliability Design

## Intent

Make grounded answers more reliable without reducing retrieval recall or reintroducing brittle lexical rejection.

## Decisions

- Keep deterministic policy responses for greetings, booking, personal medical advice, and genuinely missing evidence.
- Inspect Responses API completion state before JSON parsing and retry one incomplete structured response once with a larger output budget.
- Re-enable grounding by default using only exact citation IDs, package isolation, numeric/price consistency, and unsafe-medical-claim checks. Normal paraphrases must not be rejected for wording differences.
- Pass the detected package through runtime wrappers into validation.
- Make evidence chunk and character limits use validated settings rather than duplicate hard-coded values.
- Preserve FlashRank scores on returned results for observability and future evaluated cutoffs. Do not add a similarity cutoff in this change because the current corpus has no calibrated threshold.

## Non-goals

- No extractive answer fallback.
- No new timeout.
- No replacement vector database.
- No relevance threshold without an evaluation set large enough to calibrate it.

## Success

Existing answer and retrieval tests remain green; supported paraphrases pass; invented numbers, unknown citations, cross-package citations, and unsafe medical claims fail; incomplete structured output receives exactly one recovery attempt.
