"""Stage 4 - normalize OCR text and create tokenizer-sized chunks."""
from __future__ import annotations

import re
import unicodedata

from ..contracts import Chunk


def _normalize_text(text: str, aliases: dict) -> str:
    """Normalize OCR artifacts while preserving paragraph boundaries."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
    normalized = "".join(
        character
        for character in normalized
        if character in {"\n", "\t"} or unicodedata.category(character) != "Cc"
    )
    normalized = "\n".join(
        re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines()
    )
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()

    for remedy_id, variants in aliases.items():
        marker = f"[remedy:{remedy_id}]"
        for variant in sorted((str(value) for value in variants), key=len, reverse=True):
            if not variant:
                continue
            pattern = re.compile(re.escape(variant), flags=re.IGNORECASE)

            def add_marker(
                match: re.Match[str],
                source_text: str = normalized,
                expected_marker: str = marker,
            ) -> str:
                following = source_text[match.end() : match.end() + len(expected_marker) + 1]
                if expected_marker in following:
                    return match.group(0)
                return f"{match.group(0)} {expected_marker}"

            normalized = pattern.sub(add_marker, normalized)
    return normalized


def _entry_paragraphs(text: str) -> list[str]:
    """Group paragraphs so remedy entries are not mixed together."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return []

    entries: list[str] = []
    current: list[str] = []
    for paragraph in paragraphs:
        if "[remedy:" in paragraph and current:
            entries.append("\n\n".join(current))
            current = []
        current.append(paragraph)
    if current:
        entries.append("\n\n".join(current))
    return entries


def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Normalize OCR text, preserve remedy entries, and apply token windows."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Chunking requires transformers for tokenizer-aware windows.") from exc

    index_cfg = cfg.get("index", {})
    max_tokens = int(index_cfg.get("chunk_tokens", 512))
    overlap = int(index_cfg.get("overlap", 64))
    if max_tokens <= 0 or overlap < 0 or overlap >= max_tokens:
        raise ValueError("index.chunk_tokens must be positive and overlap must be smaller")

    model_name = str(
        cfg.get("embed", {}).get("model", "BAAI/bge-m3")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    aliases = cfg.get("normalization", {}).get("remedy_aliases", {})

    output: list[Chunk] = []
    for source in chunks:
        normalized = _normalize_text(source.text, aliases)
        if not normalized:
            continue

        pieces: list[str] = []
        for entry in _entry_paragraphs(normalized):
            token_ids = tokenizer.encode(entry, add_special_tokens=False)
            if len(token_ids) <= max_tokens:
                pieces.append(entry)
                continue

            step = max_tokens - overlap
            for start in range(0, len(token_ids), step):
                window = token_ids[start : start + max_tokens]
                decoded = tokenizer.decode(
                    window,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                if decoded:
                    pieces.append(decoded)
                if start + max_tokens >= len(token_ids):
                    break

        # Region IDs avoid duplicate chunk IDs for multiple regions on one page.
        base_id = source.id.removesuffix("_ocr")
        for position, text in enumerate(pieces, start=1):
            output.append(
                Chunk(
                    id=f"{base_id}_c{position:03d}",
                    doc_id=source.doc_id,
                    text=text,
                    page_ids=list(source.page_ids),
                    score=source.score,
                )
            )
    return output
