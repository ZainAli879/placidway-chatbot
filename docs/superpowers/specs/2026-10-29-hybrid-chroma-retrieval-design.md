# Hybrid Chroma Retrieval Design

## Intent

Restore genuine semantic first-stage retrieval without losing exact keyword recall or the current grounded-answer safeguards. The deployed Streamlit application must combine local BM25 and local persistent Chroma results, rerank the merged pool, and send only the best three to five chunks to generation.

## Goals

- Retrieve 30 BM25 candidates and 30 dense Chroma candidates for every supported knowledge question.
- Fuse rankings with reciprocal-rank fusion (RRF), deduplicate by chunk ID, and retain whether each result came from lexical, vector, or both retrieval paths.
- Allow a semantically retrieved result to reach reranking even when it has no lexical token overlap.
- Preserve named-package isolation, source approval, grounding, citation, numeric, and medical-safety behavior.
- Rerank the merged candidate pool with the existing FlashRank cross-encoder.
- Select three chunks for focused facts, four for list/coverage questions, and five for comparisons or explicitly multi-part questions.
- Build Chroma offline and reject missing or stale vector artifacts at startup; never create embeddings in a visitor request except the query embedding.
- Keep Chroma local and cached within the Streamlit process; no external vector service is required.

## Architecture

`HybridKnowledgeSearch` owns the existing `KnowledgeSearch` BM25 retriever and a persistent Chroma collection. Both legs retrieve independently. RRF combines rank positions rather than incomparable BM25 and cosine scores. `SearchResult` carries immutable retrieval provenance so the answer engine accepts dense-only candidates before FlashRank. The existing package/source priority and package isolation remain downstream safeguards.

The offline builder recreates the collection from the reviewed `chunks.jsonl`, stores chunk metadata and documents, and records the exact corpus SHA-256 plus embedding model identifier in collection metadata. Runtime validates digest, count, and chunk IDs before serving. Chroma's local default ONNX MiniLM embedding function is used so the implementation does not add PyTorch or Sentence Transformers.

## Evidence selection

- Focused fact, price, duration, location: top 3 reranked chunks.
- List, inclusions, procedures, videos: top 4 reranked chunks.
- Explicit comparison, contrast, differences, or multi-part request: top 5 reranked chunks.
- The generator remains capped at four chunks internally today and must be updated to accept the selected maximum of five while retaining the 6,000-character evidence budget.

## Failure behavior

- Missing, corrupt, or stale Chroma artifacts fail startup with a safe configuration message instead of silently degrading to BM25-only behavior.
- Query-time Chroma errors are surfaced through existing privacy-safe error handling; no question, evidence, API key, or exception message is logged.
- No deterministic factual-answer fallback is introduced.

## Evaluation

Add a small retrieval evaluation fixture containing representative exact-term, paraphrased, list, and package-isolation questions with expected chunk IDs. Report Recall@3, Recall@5, and reciprocal rank for BM25-only versus hybrid retrieval. Hybrid must not reduce the named-package cases and must recover at least one semantic-only fixture missed by BM25.

## Deployment

Commit the reviewed Chroma persistence directory together with its vector manifest. Document the offline build command. Cache the loaded hybrid searcher and query embedding function through the existing Streamlit engine cache. Update CI and dependency checks to require, rather than forbid, the Chroma runtime.

## Non-goals

- No Qdrant, LanceDB, FAISS, hosted vector service, HyDE, SPLADE, or ColBERT in this change.
- No runtime crawling or document embedding rebuild.
- No changes to generation model, response schema, greeting, booking, medical, or unrelated-query policies.

