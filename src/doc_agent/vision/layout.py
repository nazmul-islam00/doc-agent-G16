"""Stage 2 — layout detection / segmentation"""
from __future__ import annotations
import os
from PIL import Image
from surya.inference import SuryaInferenceManager
from surya.layout import LayoutPredictor
from ..contracts import *  # noqa

_MANAGER_INSTANCE = None

def _get_manager() -> SuryaInferenceManager:
    global _MANAGER_INSTANCE
    if _MANAGER_INSTANCE is None:
        _MANAGER_INSTANCE = SuryaInferenceManager()
    return _MANAGER_INSTANCE

def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect text/table/figure/heading regions. IMPLEMENT."""
    if not pages:
        return []

    # Store page metadata in cfg so ocr.py can access it without violating strict Pydantic contracts
    if "page_map" not in cfg:
        cfg["page_map"] = {}

    manager = _get_manager()
    layout_predictor = LayoutPredictor(manager)
    
    all_regions: list[Region] = []
    
    for page in pages:
        # Cache the metadata safely
        cfg["page_map"][page.id] = {"doc_id": page.doc_id, "image_path": page.image_path}
        
        if not os.path.exists(page.image_path):
            continue
            
        image = Image.open(page.image_path).convert("RGB")
        layout_preds = layout_predictor([image])
        pred = layout_preds[0]
        
        # 1. Calculate dynamic horizontal midpoint for two-column sorting
        valid_bboxes = [b.bbox for b in pred.bboxes if b.bbox]
        if valid_bboxes:
            all_x_mins = [b[0] for b in valid_bboxes]
            all_x_maxs = [b[2] for b in valid_bboxes]
            page_midpoint = (min(all_x_mins) + max(all_x_maxs)) / 2.0
        else:
            page_midpoint = image.width / 2.0

        left_column, right_column = [], []
        
        # 2. Categorize blocks
        for bbox_obj in pred.bboxes:
            if not bbox_obj.bbox:
                continue
            x_min, y_min, x_max, y_max = bbox_obj.bbox
            center_x = (x_min + x_max) / 2.0
            
            if center_x < page_midpoint:
                left_column.append(bbox_obj)
            else:
                right_column.append(bbox_obj)
                
        # 3. Sort vertically top-to-bottom
        left_column.sort(key=lambda b: b.bbox[1])
        right_column.sort(key=lambda b: b.bbox[1])
        sorted_bboxes = left_column + right_column
        
        # 4. Map to strict Region contracts
        for bbox_obj in sorted_bboxes:
            raw_label = getattr(bbox_obj, "label", "Text").lower()
            if "header" in raw_label or "title" in raw_label:
                kind = "heading"
            elif "table" in raw_label:
                kind = "table"
            elif "figure" in raw_label or "image" in raw_label:
                kind = "figure"
            else:
                kind = "text"

            # Cast Surya's floating point coordinates to exact integers (tuple)
            clean_bbox = (
                int(round(bbox_obj.bbox[0])),
                int(round(bbox_obj.bbox[1])),
                int(round(bbox_obj.bbox[2])),
                int(round(bbox_obj.bbox[3]))
            )

            # Strictly use only the fields defined in contracts.py
            region = Region(
                page_id=page.id,
                bbox=clean_bbox,
                kind=kind
            )
            all_regions.append(region)
            
    return all_regions