"""Build a transparent JSONL index from audited source text."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from app.config.sources import APPROVED_SOURCES
from app.knowledge.artifact import build_manifest
from app.knowledge.schemas import KnowledgeChunk, PriceMention, SourceDocument

AUDIT_PATH = Path("data/source_audit.json")
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def _audit_metadata() -> dict[str, dict[str, str]]:
    if not AUDIT_PATH.exists():
        raise FileNotFoundError("Run python -m scripts.audit_sources first.")
    payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    result = {}
    for row in payload.get("results", []):
        if row.get("http_status") == 200 and row.get("robots_allowed") is True:
            result[row["source_id"]] = {
                "title": row.get("title") or row["source_id"],
                "fetched_at_utc": row.get("checked_at_utc") or payload.get("generated_at_utc", ""),
                "url": row["url"],
            }
    return result


def clean_source_text(text: str) -> str:
    """Remove residual interface noise and normalize crawled page text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    cleaned = []
    ui_line = re.compile(
        r"^(skip to main content|visit placidway|about this assistant|"
        r"request a video call|select (date|time|time zone)|confirm your virtual consultation|"
        r"book a virtual consultation)$",
        re.IGNORECASE,
    )
    timezone_line = re.compile(
        r"^(midway island|samoa|pago pago|hawaii|alaska|vancouver|"
        r"pacific time|mountain time|central time|eastern time|"
        r"greenwich mean time|universal time utc|london|lisbon|tokyo|"
        r"mumbai|sydney|auckland)( \(.*\))?$",
        re.IGNORECASE,
    )
    for line in lines:
        if not line or ui_line.match(line) or timezone_line.match(line):
            continue
        if re.fullmatch(r"[|•·\-_=]{2,}", line):
            continue
        cleaned.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def load_documents() -> list[SourceDocument]:
    audit = _audit_metadata()
    sources = {s.source_id: s for s in APPROVED_SOURCES}
    docs = []
    for source_id, meta in audit.items():
        path = RAW_DIR / f"{source_id}.txt"
        if source_id not in sources or not path.is_file():
            continue
        text = clean_source_text(path.read_text(encoding="utf-8"))
        if text:
            docs.append(
                SourceDocument(
                    source_id, sources[source_id].page_type, meta["url"],
                    meta["title"], meta["fetched_at_utc"], text
                )
            )
    return docs


def _split_oversized_block(text: str, limit: int) -> list[str]:
    """Split long text on paragraph, sentence, then word boundaries."""
    if len(text) <= limit:
        return [text]

    # Keep sentence punctuation attached. Avoid arbitrary character cuts unless
    # a single token itself exceeds the configured limit.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) > 1:
        parts: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > limit:
                if current:
                    parts.append(current)
                    current = ""
                parts.extend(_split_oversized_block(sentence, limit))
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    parts.append(current)
                current = sentence
        if current:
            parts.append(current)
        if all(len(part) <= limit for part in parts):
            return parts

    words = text.split()
    parts, current = [], ""
    for word in words:
        if len(word) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.extend(word[i:i + limit] for i in range(0, len(word), limit))
            continue
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = word
    if current:
        parts.append(current)
    return parts


def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 180) -> list[str]:
    """Create bounded chunks that preserve headings and sentence boundaries."""
    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    normalized = clean_source_text(text)
    if not normalized:
        return []

    # Crawled source files usually place each DOM text line on its own line, not
    # in blank-line-delimited paragraphs. Preserve those boundaries for packing.
    blocks = [block.strip() for block in re.split(r"\n+", normalized) if block.strip()]
    heading_pattern = re.compile(
        r"^(?:#{1,6}\s+|.{1,100}:$|(?:overview|pricing|treatment|program|inpatient|outpatient|"
        r"what is included|patient suitability|itinerary|follow-up|additional costs|"
        r"package inclusions|frequently asked questions)\s*$)",
        re.IGNORECASE,
    )

    # Attach heading-only paragraphs to the following paragraph before splitting.
    sections: list[str] = []
    pending_heading = ""
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) == 1 and heading_pattern.match(lines[0]):
            pending_heading = f"{pending_heading}\n{lines[0]}".strip()
            continue
        if pending_heading:
            block = f"{pending_heading}\n{block}"
            pending_heading = ""
        sections.extend(_split_oversized_block(block, chunk_size))
    if pending_heading:
        sections.extend(_split_oversized_block(pending_heading, chunk_size))

    # Pack adjacent sections, but do not prepend a raw character tail: that was
    # creating chunks that started in the middle of a word or sentence.
    chunks: list[str] = []
    current = ""
    for section in sections:
        candidate = f"{current}\n\n{section}".strip() if current else section
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        # Keep overlap as a complete trailing sentence rather than an arbitrary
        # character slice that could start the next chunk mid-word.
        prefix = ""
        if current and overlap:
            sentences = re.split(r"(?<=[.!?])\s+", current.strip())
            if sentences:
                candidate_prefix = sentences[-1].strip()
                if candidate_prefix and len(candidate_prefix) < overlap:
                    prefix = candidate_prefix
        current = f"{prefix}\n\n{section}".strip() if prefix and len(prefix) + len(section) + 2 <= chunk_size else section
        if len(current) > chunk_size:
            pieces = _split_oversized_block(current, chunk_size)
            chunks.extend(pieces[:-1])
            current = pieces[-1]
    if current:
        chunks.append(current)

    unique: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk = chunk.strip()
        key = re.sub(r"\s+", " ", chunk).casefold()
        if chunk and key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


PRICE_PATTERN = re.compile(
    r"(?P<currency>US\s*\$|USD|\$)\s*(?P<amount>[0-9][0-9,]*(?:\.\d{1,2})?)"
    r"|(?P<amount2>[0-9][0-9,]*(?:\.\d{1,2})?)\s*(?P<currency2>USD)",
    re.IGNORECASE,
)


def extract_price_mentions(document: SourceDocument) -> list[PriceMention]:
    """Extract candidate price mentions; they require manual validation."""
    mentions = []
    for match in PRICE_PATTERN.finditer(document.text):
        amount = match.group("amount") or match.group("amount2")
        currency = match.group("currency") or match.group("currency2") or "USD"
        context = re.sub(
            r"\s+", " ",
            document.text[max(0, match.start() - 180):min(len(document.text), match.end() + 180)],
        ).strip()
        mentions.append(
            PriceMention(
                document.source_id,
                document.page_type,
                document.url,
                amount.replace(",", ""),
                "USD" if "$" in currency or currency.upper().replace(" ", "") == "USD" else currency.upper(),
                context,
                True,
            )
        )
    return mentions


def build_index(*, chunk_size: int = 1200, overlap: int = 180):
    """Build JSONL chunks, candidate price mentions, and a manifest."""
    documents = load_documents()
    if not documents:
        raise RuntimeError("No source text found. Run the audit and check data/raw/*.txt.")

    chunks, prices = [], []
    for doc in documents:
        for index, text in enumerate(chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)):
            digest = hashlib.sha256(f"{doc.source_id}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
            chunks.append(
                KnowledgeChunk(
                    f"{doc.source_id}-{digest}",
                    doc.source_id,
                    doc.page_type,
                    doc.url,
                    doc.title,
                    index,
                    text,
                )
            )
        prices.extend(extract_price_mentions(doc))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "chunks.jsonl").write_text(
        "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in chunks),
        encoding="utf-8",
    )
    (PROCESSED_DIR / "price_mentions_review.jsonl").write_text(
        "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in prices),
        encoding="utf-8",
    )
    corpus_path = PROCESSED_DIR / "chunks.jsonl"
    manifest = build_manifest(
        corpus_path,
        [item.to_dict() for item in chunks],
        built_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    manifest.update(
        documents_loaded=len(documents),
        candidate_price_mentions=len(prices),
        price_mentions_are_verified=False,
    )
    (PROCESSED_DIR / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return chunks, prices


if __name__ == "__main__":
    chunks, prices = build_index()
    print(f"Built {len(chunks)} chunks; extracted {len(prices)} candidate price mentions for review.")
    print("Review data/processed/price_mentions_review.jsonl before using prices in answers.")
