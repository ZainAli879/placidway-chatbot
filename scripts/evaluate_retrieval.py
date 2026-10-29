"""Evaluate lexical or deployed hybrid retrieval against reviewed cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.retrieval.evaluation import RetrievalCase, evaluate_retrieval
from app.retrieval.hybrid_search import HybridKnowledgeSearch
from app.retrieval.search import KnowledgeSearch


def _load_cases(path: Path) -> list[RetrievalCase]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        RetrievalCase(row["query"], frozenset(row["relevant_chunk_ids"]))
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bm25", "hybrid"), default="hybrid")
    parser.add_argument("--index", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--chroma", type=Path, default=Path("data/processed/chroma"))
    parser.add_argument("--cases", type=Path, default=Path("data/evaluation/retrieval_cases.json"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "hybrid":
        searcher = HybridKnowledgeSearch.from_jsonl(
            args.index, persist_directory=args.chroma
        )
    else:
        searcher = KnowledgeSearch.from_jsonl(args.index)

    metrics = evaluate_retrieval(searcher, _load_cases(args.cases), top_k=args.top_k)
    print(json.dumps({"mode": args.mode, **metrics}, indent=2))


if __name__ == "__main__":
    main()
