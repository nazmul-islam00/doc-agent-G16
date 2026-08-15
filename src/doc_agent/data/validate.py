"""Data -- page-level schema and quality validation at ingest."""

from __future__ import annotations

import re
from pathlib import Path

from .. import config
from ..contracts import Page

_PAGE_PATTERN = re.compile(r"page-(\d+)$")
_SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _configured_chapter_splits(data_cfg: dict) -> tuple[dict[int, int], dict[int, str]]:
    """Validate configured ranges/splits and return page and chapter ownership."""
    configured_ranges = data_cfg.get("chapter_ranges", {})
    if not isinstance(configured_ranges, dict) or not configured_ranges:
        raise ValueError("data.chapter_ranges must define at least one chapter range.")

    page_to_chapter: dict[int, int] = {}
    for raw_chapter, raw_range in configured_ranges.items():
        chapter = int(raw_chapter)
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            raise ValueError(f"Chapter {chapter} range must be a [start, end] pair.")
        start, end = (int(value) for value in raw_range)
        if chapter <= 0 or start <= 0 or start > end:
            raise ValueError(f"Chapter {chapter} has invalid range [{start}, {end}].")
        for page_number in range(start, end + 1):
            if page_number in page_to_chapter:
                other_chapter = page_to_chapter[page_number]
                raise ValueError(
                    f"Page {page_number} belongs to chapters {other_chapter} and {chapter}."
                )
            page_to_chapter[page_number] = chapter

    configured_splits = data_cfg.get("chapter_splits", {})
    required_splits = {"train", "validation", "test"}
    if not isinstance(configured_splits, dict) or not required_splits.issubset(configured_splits):
        raise ValueError("data.chapter_splits must define train, validation, and test chapters.")

    chapter_to_split: dict[int, str] = {}
    for split in sorted(required_splits):
        chapters = configured_splits[split]
        if not isinstance(chapters, list):
            raise ValueError(f"data.chapter_splits.{split} must be a list.")
        for raw_chapter in chapters:
            chapter = int(raw_chapter)
            if chapter not in set(page_to_chapter.values()):
                raise ValueError(f"Split {split!r} references unknown chapter {chapter}.")
            if chapter in chapter_to_split:
                other_split = chapter_to_split[chapter]
                raise ValueError(
                    f"Chapter {chapter} belongs to both {other_split!r} and {split!r} splits."
                )
            chapter_to_split[chapter] = split

    configured_chapters = set(page_to_chapter.values())
    assigned_chapters = set(chapter_to_split)
    if configured_chapters != assigned_chapters:
        missing_chapters = sorted(configured_chapters - assigned_chapters)
        raise ValueError(f"Chapter splits omit chapters: {missing_chapters}.")
    return page_to_chapter, chapter_to_split


def _page_number(page_id: str) -> int | None:
    """Extract a rendered PDF page number from a stable page ID."""
    match = _PAGE_PATTERN.fullmatch(Path(page_id).name)
    return int(match.group(1)) if match is not None else None


def validate(pages: list[Page]) -> None:
    """Validate loaded image pages and the configured chapter-grouped split.

    OCR word-count validation must occur after OCR because the fixed ``Page``
    contract contains image metadata, not text.
    """
    task_cfg = config.load_task()
    data_cfg = config.load().get("data", {})
    minimum_pages = int(task_cfg.get("corpus", {}).get("min_pages", 300))
    if minimum_pages <= 0:
        raise ValueError("Configured corpus.min_pages must be positive.")
    if len(pages) < minimum_pages:
        raise ValueError(f"Corpus has {len(pages)} pages; at least {minimum_pages} are required.")

    page_to_chapter: dict[int, int] = {}
    if bool(data_cfg.get("chapter_pages_only", False)):
        page_to_chapter, _ = _configured_chapter_splits(data_cfg)

    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for position, page in enumerate(pages, start=1):
        if not isinstance(page, Page):
            raise TypeError(f"Page {position} must be a Page instance, got {type(page).__name__}.")
        if not page.id.strip():
            raise ValueError(f"Page {position} has an empty page ID.")
        if page.id in seen_ids:
            raise ValueError(f"Duplicate page ID: {page.id!r}.")
        seen_ids.add(page.id)

        if not page.doc_id.strip():
            raise ValueError(f"Page {page.id!r} has an empty document ID.")

        image_path = Path(page.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Page {page.id!r} points to a missing image: {image_path}")
        if image_path.suffix.lower() not in _SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Page {page.id!r} has unsupported image type: {image_path.suffix!r}.")

        resolved_path = image_path.resolve()
        if resolved_path in seen_paths:
            raise ValueError(f"Duplicate image path in corpus: {resolved_path}")
        seen_paths.add(resolved_path)

        if page_to_chapter:
            page_number = _page_number(page.id)
            if page_number not in page_to_chapter:
                raise ValueError(f"Page {page.id!r} is not in a configured chapter range.")
