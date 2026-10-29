"""Validate the production knowledge artifact and exit non-zero on failure."""
from __future__ import annotations

import argparse
from pathlib import Path

from app.knowledge.artifact import validate_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/index_manifest.json"))
    args = parser.parse_args()
    metadata = validate_artifact(args.corpus, args.manifest)
    print(f"Valid reviewed artifact: {metadata.chunk_count} chunks, {metadata.source_count} sources")


if __name__ == "__main__":
    main()
