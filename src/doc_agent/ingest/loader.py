"""Stage 1 — load scanned page-images"""

from __future__ import annotations

import re
from pathlib import Path

from ..contracts import Page

_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_KRISHIPATH_CHAPTER_RANGES = (
    (14, 19),
    (20, 25),
    (26, 28),
    (29, 29),
    (30, 34),
    (35, 55),
    (56, 122),
    (123, 135),
    (136, 138),
    (139, 144),
    (145, 147),
    (148, 156),
    (157, 159),
    (160, 168),
    (169, 175),
    (176, 230),
    (231, 249),
    (250, 256),
    (257, 262),
    (263, 266),
    (267, 269),
    (270, 317),
    (318, 323),
    (324, 327),
    (328, 346),
    (347, 351),
    (352, 355),
    (356, 364),
    (365, 374),
    (375, 382),
    (383, 385),
    (386, 389),
    (390, 391),
)
_KRISHIPATH_PAGE_PATTERN = re.compile(r"page-(\d+)$")


def _is_krishipath_page(relative_path: Path, raw_dir: Path) -> bool:
    """Identify the project's rendered Krishipath page-image naming scheme."""
    return raw_dir.name == "krishipath" or (
        bool(relative_path.parts) and relative_path.parts[0] == "krishipath"
    )


def _is_chapter_page(path: Path) -> bool:
    """Return whether a rendered Krishipath page belongs to one of 33 chapters."""
    match = _KRISHIPATH_PAGE_PATTERN.fullmatch(path.stem)
    if match is None:
        return False
    page_number = int(match.group(1))
    return any(start <= page_number <= end for start, end in _KRISHIPATH_CHAPTER_RANGES)


def load_pages(cfg: dict) -> list[Page]:
    """Load scanned page images from the configured raw-data directory.

    Images may be stored directly in ``data/raw`` or grouped in subdirectories by
    document. Page IDs are relative paths without extensions, which keeps them
    stable even when the corpus contains repeated filenames in different folders.

    Krishipath has 378 chapter pages and 25 auxiliary pages (cover, contents,
    bibliography, and similar matter). By default, its auxiliary pages are
    excluded from the knowledge-base corpus. Set ``data.chapter_pages_only`` to
    ``false`` to deliberately include all 403 rendered pages.
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
    chapter_pages_only = data_cfg.get("chapter_pages_only")
    if chapter_pages_only is not False:
        image_paths = [
            path
            for path in image_paths
            if not _is_krishipath_page(path.relative_to(raw_dir), raw_dir) or _is_chapter_page(path)
        ]

    if not image_paths:
        raise ValueError(f"No supported page images found in {raw_dir}")

    pages: list[Page] = []
    for path in image_paths:
        relative_path = path.relative_to(raw_dir)
        page_id = relative_path.with_suffix("").as_posix()
        doc_id = relative_path.parent.as_posix() if relative_path.parent != Path(".") else "default"
        pages.append(Page(id=page_id, image_path=str(path), doc_id=doc_id))

    return pages
