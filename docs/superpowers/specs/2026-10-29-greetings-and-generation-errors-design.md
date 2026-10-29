# Greetings and Generation Error Diagnostics

## Intent

Make standalone greetings feel professional and immediate, while making generation failures diagnosable instead of collapsing every exception into an unexplained unavailable result.

## Scope

- Recognize standalone greetings such as `hi`, `hello`, `hey`, and time-of-day greetings, including normal punctuation and polite greeting suffixes.
- Return a concise professional welcome without retrieval or model generation.
- Do not consume a greeting that also contains a substantive question; it must continue through the existing grounded pipeline.
- Preserve a privacy-safe generation failure category on `Answer` and emit it through runtime telemetry.
- Distinguish provider/API errors, empty output, invalid structured output, and grounding validation rejection.

## Non-goals

- No changes to booking, medical, weather, unrelated-query, retrieval, reranking, or evidence-selection behavior.
- No deterministic factual answer fallback.
- No logging of prompts, evidence text, API keys, exception messages, or sensitive configuration.

## Approach

Use a narrow local greeting classifier before retrieval because greetings need no page evidence and should be instant. Introduce typed generation exceptions at the generator boundary and carry only their safe category through `Answer` to Streamlit telemetry. Unknown provider exceptions retain their exception class name for diagnosis without logging their message.

