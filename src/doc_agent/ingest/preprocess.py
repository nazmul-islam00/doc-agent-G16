"""Stage 1 -- page preprocessing before layout detection and OCR."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..contracts import Page

LOGGER = logging.getLogger(__name__)


def _estimate_text_line_skew(image: np.ndarray) -> float | None:
    """Estimate skew from long, near-horizontal text lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    _, width = gray.shape
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=max(35, width // 18),
        minLineLength=max(80, width // 7),
        maxLineGap=max(8, width // 120),
    )
    if lines is None:
        return None

    angles: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if math.hypot(float(x2 - x1), float(y2 - y1)) < width * 0.12:
            continue
        angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
        if abs(angle) <= 2.0:
            angles.append(angle)
    if len(angles) < 6:
        return None

    median = float(np.median(angles))
    inliers = [angle for angle in angles if abs(angle - median) <= 0.25]
    return float(np.median(inliers)) if len(inliers) >= 6 else None


def _trim(image: np.ndarray, trim_left_px: int, trim_border_px: int) -> np.ndarray:
    """Apply configured edge trims."""
    height, width = image.shape[:2]
    left = trim_left_px + trim_border_px
    top = trim_border_px
    right = width - trim_border_px
    bottom = height - trim_border_px
    if left >= right - 16 or top >= bottom - 16:
        raise ValueError("Configured preprocessing trim leaves an invalid image.")
    return image[top:bottom, left:right]


def _preprocess_image(image: np.ndarray, settings: dict[str, float | int | bool]) -> np.ndarray:
    """Apply configured edge trimming and deskewing."""
    image = _trim(image, int(settings["trim_left_px"]), int(settings["trim_border_px"]))
    if not bool(settings["deskew"]):
        return image

    skew = _estimate_text_line_skew(image)
    if skew is None:
        return image
    if not float(settings["min_skew_degrees"]) <= abs(skew) <= float(settings["max_skew_degrees"]):
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -skew, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _settings(cfg: dict) -> dict[str, float | int | bool | Path]:
    configured = cfg.get("preprocess", {})
    return {
        "enabled": bool(configured.get("enabled", True)),
        "output_dir": Path(configured.get("output_dir", "data/interim/preprocessed")),
        "trim_left_px": max(0, int(configured.get("trim_left_px", 80))),
        "trim_border_px": max(0, int(configured.get("trim_border_px", 0))),
        "deskew": bool(configured.get("deskew", True)),
        "min_skew_degrees": max(0.0, float(configured.get("min_skew_degrees", 0.15))),
        "max_skew_degrees": max(0.0, float(configured.get("max_skew_degrees", 2.0))),
    }


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Write preprocessed images without modifying the raw corpus."""
    settings = _settings(cfg)
    if not bool(settings["enabled"]):
        return pages

    output_dir = Path(settings["output_dir"])
    processed_pages: list[Page] = []
    for page in pages:
        source_path = Path(page.image_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Page image does not exist: {source_path}")
        with Image.open(source_path) as source:
            image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        output_path = output_dir / Path(page.id).with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_preprocess_image(image, settings)).save(
            output_path, format="PNG", optimize=True
        )
        processed_pages.append(page.model_copy(update={"image_path": str(output_path)}))

    LOGGER.info("Preprocessed %d page(s) into %s", len(processed_pages), output_dir)
    return processed_pages
