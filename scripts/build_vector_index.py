"""Build the persistent semantic index from the reviewed chunk corpus."""
from __future__ import annotations

import argparse

from app.retrieval.hybrid_search import build_vector_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/processed/chunks.jsonl")
    parser.add_argument("--output", default="data/processed/chroma")
    args = parser.parse_args()
    collection = build_vector_index(args.corpus, persist_directory=args.output)
    print(f"Built {collection.count()} persistent semantic vectors in {args.output}.")


if __name__ == "__main__":
    main()
