"""Stage 1 -- conservative classical cleanup of scanned page images.

The raw corpus is never modified. This stage writes derived PNGs and returns
``Page`` objects which retain their original stable IDs and document IDs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..contracts import Page

LOGGER = logging.getLogger(__name__)


def _rotate(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate around the image centre, filling exposed corners with white."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_degrees, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _estimate_skew_degrees(gray: np.ndarray) -> float | None:
    """Estimate text-block skew using dark foreground pixels.

    A narrow frame is excluded so scanner borders and page shadows cannot
    dominate the estimate. ``None`` means the page lacks usable foreground.
    """
    height, width = gray.shape
    pad_y = max(1, round(height * 0.03))
    pad_x = max(1, round(width * 0.03))
    centre = gray[pad_y : height - pad_y, pad_x : width - pad_x]
    if centre.size == 0:
        return None

    _, foreground = cv2.threshold(centre, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coordinates = cv2.findNonZero(foreground)
    if coordinates is None or len(coordinates) < 100:
        return None

    raw_angle = float(cv2.minAreaRect(coordinates)[-1])
    # OpenCV's rectangle-angle range differs between builds. This maps it to
    # the closest horizontal orientation, ready to be negated for correction.
    return raw_angle - 90.0 if raw_angle > 45.0 else raw_angle


def _detect_page_crop(gray: np.ndarray, min_area_ratio: float) -> tuple[int, int, int, int] | None:
    """Return a likely physical-page bounding rectangle, or ``None`` safely."""
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=7, sigmaY=7)
    _, paper = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_size = max(9, (min(height, width) // 40) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, crop_width, crop_height = cv2.boundingRect(contour)
    if crop_width * crop_height < height * width * min_area_ratio:
        return None
    if crop_width < width * 0.4 or crop_height < height * 0.4:
        return None
    return x, y, crop_width, crop_height


def _trim_border(image: np.ndarray, margin: int) -> np.ndarray:
    """Remove a small post-crop border without allowing an invalid image."""
    if margin <= 0:
        return image
    height, width = image.shape[:2]
    if width <= margin * 2 + 16 or height <= margin * 2 + 16:
        return image
    return image[margin : height - margin, margin : width - margin]


def _normalize_background(gray: np.ndarray, paper_min_median: float) -> np.ndarray:
    """Flatten uneven/yellow paper illumination while keeping fine ink strokes.

    Dark covers and illustration pages are not paper-like, so background
    division would invert or flatten their useful colour contrast. They are
    deliberately left as ordinary grayscale pages.
    """
    if float(np.median(gray)) < paper_min_median:
        return gray
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
    background = np.maximum(background, 1)
    normalized = cv2.divide(gray, background, scale=255)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)


def _preprocess_image(image: np.ndarray, settings: dict) -> np.ndarray:
    """Clean one RGB page using only operations explicitly enabled in config."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if settings["deskew"]:
        skew = _estimate_skew_degrees(gray)
        if skew is not None and abs(skew) <= settings["max_skew_degrees"]:
            image = _rotate(image, -skew)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if settings["crop_page"]:
        crop = _detect_page_crop(gray, settings["min_page_area_ratio"])
        if crop is not None:
            x, y, width, height = crop
            image = image[y : y + height, x : x + width]

    image = _trim_border(image, settings["border_margin_px"])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if settings["denoise"]:
        gray = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
    if settings["normalize_background"]:
        gray = _normalize_background(gray, settings["paper_min_median"])
    if settings["binarize"]:
        if settings["binarize_method"] == "otsu":
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                35,
                11,
            )

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _settings(cfg: dict) -> dict:
    configured = cfg.get("preprocess", {})
    return {
        "enabled": bool(configured.get("enabled", True)),
        "output_dir": Path(configured.get("output_dir", "data/interim/preprocessed")),
        "deskew": bool(configured.get("deskew", True)),
        "max_skew_degrees": float(configured.get("max_skew_degrees", 3.0)),
        "crop_page": bool(configured.get("crop_page", True)),
        "border_margin_px": max(0, int(configured.get("border_margin_px", 12))),
        "min_page_area_ratio": float(configured.get("min_page_area_ratio", 0.55)),
        "denoise": bool(configured.get("denoise", True)),
        "normalize_background": bool(configured.get("normalize_background", True)),
        "paper_min_median": float(configured.get("paper_min_median", 160.0)),
        "binarize": bool(configured.get("binarize", False)),
        "binarize_method": str(configured.get("binarize_method", "adaptive")).lower(),
    }


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Create preprocessed page images and return pages pointing to them.

    Output filenames are derived from stable ``Page.id`` values, so nested
    source documents retain their directory structure and duplicate basenames
    cannot overwrite one another.
    """
    settings = _settings(cfg)
    if not settings["enabled"]:
        return pages

    output_dir = settings["output_dir"]
    processed_pages: list[Page] = []
    for page in pages:
        source_path = Path(page.image_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Page image does not exist: {source_path}")

        output_path = output_dir / Path(page.id).with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            image = np.asarray(oriented)

        processed = _preprocess_image(image, settings)
        Image.fromarray(processed).save(output_path, format="PNG", optimize=True)
        processed_pages.append(page.model_copy(update={"image_path": str(output_path)}))

    LOGGER.info("Preprocessed %d page(s) into %s", len(processed_pages), output_dir)
    return processed_pages
