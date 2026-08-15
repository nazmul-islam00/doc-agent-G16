"""Stage 3 -- full-page Bengali/English OCR with EasyOCR."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..contracts import Chunk, Region

LOGGER = logging.getLogger(__name__)


def _use_gpu(requested: object) -> bool:
    """Resolve the configured EasyOCR device."""
    requested_text = str(requested or "auto").lower()
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    if requested_text == "auto":
        return cuda_available
    if requested_text.startswith("cuda") and not cuda_available:
        LOGGER.warning("CUDA was requested for EasyOCR but is unavailable; using CPU.")
        return False
    return requested_text.startswith("cuda")


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


class Reader:
    """EasyOCR reader configured for Bengali and English."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg.get("ocr", {})
        languages = self.cfg.get("languages", ["bn", "en"])
        if not isinstance(languages, list) or not all(isinstance(item, str) for item in languages):
            raise ValueError("ocr.languages must be a list of EasyOCR language codes.")
        self.languages = languages
        self.gpu = _use_gpu(self.cfg.get("device", cfg.get("device", "auto")))
        self.min_confidence = float(self.cfg.get("min_confidence", 0.0))
        try:
            import easyocr
        except ImportError as exc:
            raise ImportError(
                "Install the vision dependencies with `uv sync` before running OCR."
            ) from exc
        self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)

    def read_page(self, image_path: str) -> list[dict[str, Any]]:
        """Read text in a page image."""
        with Image.open(image_path) as source:
            image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        lines: list[dict[str, Any]] = []
        for box, text, confidence in self._reader.readtext(image, detail=1, paragraph=False):
            cleaned = _normalise_text(str(text))
            if not cleaned or float(confidence) < self.min_confidence:
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            lines.append(
                {
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                    "text": cleaned,
                }
            )
        return sorted(lines, key=lambda line: (line["bbox"][1], line["bbox"][0]))

    def transcribe_region(self, region: Region) -> str:
        """Return region text."""
        del region
        return ""


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """OCR source pages referenced by the detected regions."""
    if not regions:
        return []
    page_map = cfg.get("_page_map", {})
    if not isinstance(page_map, dict):
        raise ValueError("Layout stage did not provide the page metadata needed for OCR.")

    reader = Reader(cfg)
    chunks: list[Chunk] = []
    for page_id in dict.fromkeys(region.page_id for region in regions):
        page_data = page_map.get(page_id)
        if not isinstance(page_data, dict):
            LOGGER.warning("No source metadata was found for page %s; skipping OCR.", page_id)
            continue
        image_path = Path(str(page_data.get("image_path", "")))
        if not image_path.is_file():
            LOGGER.warning("Source image for page %s does not exist: %s", page_id, image_path)
            continue
        text = "\n".join(line["text"] for line in reader.read_page(str(image_path)))
        text = _normalise_text(text)
        if text:
            chunks.append(
                Chunk(
                    id=f"{page_id}_ocr",
                    doc_id=str(page_data.get("doc_id", "unknown_doc")),
                    text=text,
                    page_ids=[page_id],
                )
            )
    return chunks
