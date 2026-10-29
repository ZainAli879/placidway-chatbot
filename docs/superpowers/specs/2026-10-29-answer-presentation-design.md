# Answer Presentation Design

## Intent

Make generated answers easier for visitors to scan without losing supported details from retrieved evidence.

## Selected approach

Use prompt guidance only. The model chooses Markdown paragraphs or bullets according to the answer shape. No deterministic parser, formatter, fallback, post-processing rule, or retrieval change is added.

## Requirements

- Preserve every materially relevant fact requested by the visitor when it is supported by supplied evidence.
- Never omit prices, durations, qualifiers, exceptions, inclusions, or package distinctions merely to shorten an answer.
- Use a short lead-in and Markdown bullets for multi-item inclusions, steps, options, or comparison points.
- Use a concise paragraph for a simple one-fact answer.
- Avoid repeating facts or source titles in the generated answer body.
- Keep the current strict structured output, evidence citations, grounding validation, retrieval, and reranking unchanged.

## Non-goals

- No code-based conversion of prose into lists.
- No new fallback or hard-coded answer templates.
- No reduction of the answer token budget or evidence budget.

