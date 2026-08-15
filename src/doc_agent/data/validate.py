"""Data -- page-level schema and quality validation at ingest."""
from __future__ import annotations

from pathlib import Path

from .. import config
from ..contracts import Page

_SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def validate(pages: list[Page]) -> None:
    """Validate the loaded image-page corpus before preprocessing.

    The fixed ``Page`` contract contains image metadata only. Consequently,
    OCR word-count and train/validation/test leakage checks must occur at the
    stages where text and split metadata are available.
    """
    task_cfg = config.load_task()
    minimum_pages = int(task_cfg.get("corpus", {}).get("min_pages", 300))
    if minimum_pages <= 0:
        raise ValueError("Configured corpus.min_pages must be positive.")
    if len(pages) < minimum_pages:
        raise ValueError(
            f"Corpus has {len(pages)} pages; at least {minimum_pages} are required."
        )

    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for position, page in enumerate(pages, start=1):
        if not isinstance(page, Page):
            raise TypeError(
                f"Page {position} must be a Page instance, got {type(page).__name__}."
            )
        if not page.id.strip():
            raise ValueError(f"Page {position} has an empty page ID.")
        if page.id in seen_ids:
            raise ValueError(f"Duplicate page ID: {page.id!r}.")
        seen_ids.add(page.id)

        if not page.doc_id.strip():
            raise ValueError(f"Page {page.id!r} has an empty document ID.")

        image_path = Path(page.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Page {page.id!r} points to a missing image: {image_path}"
            )
        if image_path.suffix.lower() not in _SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Page {page.id!r} has unsupported image type: {image_path.suffix!r}."
            )

        resolved_path = image_path.resolve()
        if resolved_path in seen_paths:
            raise ValueError(f"Duplicate image path in corpus: {resolved_path}")
        seen_paths.add(resolved_path)
