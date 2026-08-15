"""Stage 1 -- load scanned page images."""

from __future__ import annotations

import re
from pathlib import Path

from ..contracts import Page

_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_PAGE_PATTERN = re.compile(r"page-(\d+)$")


def _chapter_ranges(data_cfg: dict) -> dict[int, tuple[int, int]]:
    """Parse the configured chapter ranges into validated integer pairs."""
    configured_ranges = data_cfg.get("chapter_ranges", {})
    if not isinstance(configured_ranges, dict) or not configured_ranges:
        raise ValueError("data.chapter_ranges must define at least one chapter range.")

    ranges: dict[int, tuple[int, int]] = {}
    for raw_chapter, raw_range in configured_ranges.items():
        chapter = int(raw_chapter)
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            raise ValueError(f"Chapter {chapter} range must be a [start, end] pair.")
        start, end = (int(value) for value in raw_range)
        if chapter <= 0 or start <= 0 or start > end:
            raise ValueError(f"Chapter {chapter} has invalid range [{start}, {end}].")
        ranges[chapter] = (start, end)
    return ranges


def _is_chapter_page(path: Path, chapter_ranges: dict[int, tuple[int, int]]) -> bool:
    """Return whether a rendered ``page-NNNN`` image is in a chapter range."""
    match = _PAGE_PATTERN.fullmatch(path.stem)
    if match is None:
        return False
    page_number = int(match.group(1))
    return any(start <= page_number <= end for start, end in chapter_ranges.values())


def load_pages(cfg: dict) -> list[Page]:
    """Load scanned page images from the configured raw-data directory.

    Images may be stored directly in ``data/raw`` or grouped in subdirectories
    by document. Page IDs are relative paths without extensions, which keeps
    them stable even when the corpus contains repeated filenames in different
    folders. When ``data.chapter_pages_only`` is enabled, only pages matching
    the configured ``data.chapter_ranges`` are loaded.
    """
    data_cfg = cfg.get("data", {})
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw corpus directory does not exist: {raw_dir}")

    image_paths = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if bool(data_cfg.get("chapter_pages_only", False)):
        chapter_ranges = _chapter_ranges(data_cfg)
        image_paths = [path for path in image_paths if _is_chapter_page(path, chapter_ranges)]

    if not image_paths:
        raise ValueError(f"No supported page images found in {raw_dir}")

    pages: list[Page] = []
    for path in image_paths:
        relative_path = path.relative_to(raw_dir)
        page_id = relative_path.with_suffix("").as_posix()
        doc_id = relative_path.parent.as_posix() if relative_path.parent != Path(".") else "default"
        pages.append(Page(id=page_id, image_path=str(path), doc_id=doc_id))

    return pages
