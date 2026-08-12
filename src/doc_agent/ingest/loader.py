"""Stage 1 — load scanned page-images"""
from __future__ import annotations

from pathlib import Path

from ..contracts import Page

_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_pages(cfg: dict) -> list[Page]:
    """Load scanned page images from the configured raw-data directory.

    Images may be stored directly in ``data/raw`` or grouped in subdirectories by
    document. Page IDs are relative paths without extensions, which keeps them
    stable even when the corpus contains repeated filenames in different folders.
    """
    raw_dir = Path(cfg.get("data", {}).get("raw_dir", "data/raw"))
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw corpus directory does not exist: {raw_dir}")

    image_paths = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported page images found in {raw_dir}")

    pages: list[Page] = []
    for path in image_paths:
        relative_path = path.relative_to(raw_dir)
        page_id = relative_path.with_suffix("").as_posix()
        doc_id = relative_path.parent.as_posix() if relative_path.parent != Path(".") else "default"
        pages.append(Page(id=page_id, image_path=str(path), doc_id=doc_id))

    return pages
