"""Stage 2 -- document-layout detection with DocLayout-YOLO (YOLOv10)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..contracts import Page, Region

LOGGER = logging.getLogger(__name__)


def _device(requested: object) -> str:
    """Resolve the configured layout device."""
    requested_text = str(requested or "auto").lower()
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    if requested_text == "auto":
        return "cuda:0" if cuda_available else "cpu"
    if requested_text.startswith("cuda") and not cuda_available:
        LOGGER.warning("CUDA was requested for layout detection but is unavailable; using CPU.")
        return "cpu"
    return requested_text


def _weights_path(layout_cfg: dict[str, Any]) -> str:
    """Resolve the configured weight file."""
    model = str(layout_cfg.get("model", "juliozhao/DocLayout-YOLO-DocStructBench"))
    local_path = Path(model)
    if local_path.is_file():
        return str(local_path)
    if "/" not in model:
        return model

    filename = str(layout_cfg.get("weights_filename", "doclayout_yolo_docstructbench_imgsz1024.pt"))
    allow_download = bool(layout_cfg.get("allow_download", True))
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Install huggingface-hub or configure layout.model with a local YOLOv10 weight file."
        ) from exc
    try:
        return hf_hub_download(
            repo_id=model,
            filename=filename,
            local_files_only=not allow_download,
        )
    except Exception as exc:
        source = "local cache" if not allow_download else "the configured Hugging Face repository"
        raise RuntimeError(
            f"Could not load layout weights '{filename}' from {source}. "
            "Provide a local .pt file or enable model download."
        ) from exc


def _kind(label: str) -> str:
    normalized = re.sub(r"[^a-z]+", "", label.lower())
    if normalized in {"title", "sectionheader"} or "header" in normalized:
        return "heading"
    if "table" in normalized:
        return "table"
    if normalized in {"picture", "figure"}:
        return "figure"
    return "text"


def _ordered(regions: list[Region], width: int, mode: str) -> list[Region]:
    def key(region: Region) -> tuple[int, int]:
        return region.bbox[1], region.bbox[0]

    if mode != "columns":
        return sorted(regions, key=key)
    midpoint = width / 2.0
    left = [region for region in regions if (region.bbox[0] + region.bbox[2]) / 2 < midpoint]
    right = [region for region in regions if (region.bbox[0] + region.bbox[2]) / 2 >= midpoint]
    return sorted(left, key=key) + sorted(right, key=key)


def _model(layout_cfg: dict[str, Any]) -> Any:
    try:
        from doclayout_yolo import YOLOv10
    except ImportError as exc:
        raise ImportError(
            "Install the vision dependencies with `uv sync` before running layout detection."
        ) from exc
    return YOLOv10(_weights_path(layout_cfg))


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect page regions using DocLayout-YOLO (YOLOv10)."""
    if not pages:
        return []

    layout_cfg = cfg.get("layout", {})
    device = _device(layout_cfg.get("device", cfg.get("device", "auto")))
    model = _model(layout_cfg)
    confidence = float(layout_cfg.get("score_thr", 0.2))
    image_size = int(layout_cfg.get("imgsz", 1024))
    order = str(layout_cfg.get("reading_order", "top-to-bottom"))
    page_map = cfg.setdefault("_page_map", {})
    all_regions: list[Region] = []

    for page in pages:
        image_path = Path(page.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Page image does not exist: {image_path}")
        page_map[page.id] = {"doc_id": page.doc_id, "image_path": str(image_path)}
        result = model.predict(
            str(image_path),
            imgsz=image_size,
            conf=confidence,
            device=0 if device.startswith("cuda") else "cpu",
            verbose=False,
        )[0]
        image_height, image_width = result.orig_shape
        page_regions: list[Region] = []
        if result.boxes is not None:
            coordinates = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            names = result.names
            for bbox, class_id in zip(coordinates, classes, strict=True):
                left, top, right, bottom = (int(round(value)) for value in bbox)
                left, right = max(0, left), min(image_width, right)
                top, bottom = max(0, top), min(image_height, bottom)
                if right <= left or bottom <= top:
                    continue
                label = names[int(class_id)] if isinstance(names, dict) else names[int(class_id)]
                page_regions.append(
                    Region(page_id=page.id, bbox=(left, top, right, bottom), kind=_kind(str(label)))
                )
        all_regions.extend(_ordered(page_regions, image_width, order))
    return all_regions
