"""Deliberately approve a manually reviewed knowledge corpus."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from app.knowledge.artifact import approve_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/index_manifest.json"))
    args = parser.parse_args()
    approved = approve_artifact(args.corpus, args.manifest, args.reviewer, datetime.now(timezone.utc))
    print(f"Approved {approved['chunk_count']} chunks as {approved['corpus_sha256']}")


if __name__ == "__main__":
    main()
