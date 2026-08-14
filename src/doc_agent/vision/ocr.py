"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""
from __future__ import annotations
import uuid
import os
from PIL import Image
import html2text
from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor
from ..contracts import *  # noqa

_MANAGER_INSTANCE = None

def _get_manager() -> SuryaInferenceManager:
    global _MANAGER_INSTANCE
    if _MANAGER_INSTANCE is None:
        _MANAGER_INSTANCE = SuryaInferenceManager()
    return _MANAGER_INSTANCE

class Reader:
    """Model set by cfg['ocr']. Baseline: pretrained TrOCR/Donut/Tesseract."""
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg.get("ocr", {})
        self.predictor = RecognitionPredictor(_get_manager())
        self.html_parser = html2text.HTML2Text()
        self.html_parser.body_width = 0

    def transcribe_region(self, region: Region) -> str:
        """Legacy block fallback. Unused when running full-page contextual OCR."""
        return ""

def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Regions -> text chunks. IMPLEMENT (calls Reader)."""
    if not regions:
        return []

    reader = Reader(cfg)
    chunks: list[Chunk] = []
    
    # Retrieve the metadata cache passed from layout.py
    page_map = cfg.get("page_map", {})
    
    # Group regions by page to run full-page OCR instead of block OCR
    regions_by_page = set(r.page_id for r in regions)

    for page_id in regions_by_page:
        page_data = page_map.get(page_id, {})
        image_path = page_data.get("image_path")
        doc_id = page_data.get("doc_id", "unknown_doc")
        
        if not image_path or not os.path.exists(image_path):
            continue
            
        image = Image.open(image_path).convert("RGB")
        
        # Run full-page OCR to preserve global context
        ocr_preds = reader.predictor([image])
        if not ocr_preds or not ocr_preds[0].blocks:
            continue
            
        result = ocr_preds[0]
        
        processed_blocks = []
        for block in result.blocks:
            if hasattr(block, 'polygon') and block.polygon:
                xs = [pt[0] for pt in block.polygon]
                ys = [pt[1] for pt in block.polygon]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                bbox = getattr(block, 'bbox', [0,0,0,0])
            processed_blocks.append({"block": block, "bbox": bbox})
            
        # Recalculate 2-column layout natively 
        all_x_mins = [item["bbox"][0] for item in processed_blocks if item["bbox"][2] > 0]
        all_x_maxs = [item["bbox"][2] for item in processed_blocks if item["bbox"][2] > 0]
        page_midpoint = (min(all_x_mins) + max(all_x_maxs)) / 2.0 if all_x_mins else image.width / 2.0
        
        left_column, right_column = [], []
        for item in processed_blocks:
            center_x = (item["bbox"][0] + item["bbox"][2]) / 2.0
            if center_x < page_midpoint:
                left_column.append(item)
            else:
                right_column.append(item)
                
        left_column.sort(key=lambda x: x["bbox"][1])
        right_column.sort(key=lambda x: x["bbox"][1])
        sorted_items = left_column + right_column

        # Map the sorted texts to Chunk objects strictly following the contract
        for item in sorted_items:
            raw_html = item["block"].html if item["block"].html else ""
            clean_text = reader.html_parser.handle(raw_html).strip() if raw_html else ""
            
            if clean_text:
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=clean_text,
                    page_ids=[page_id]
                )
                chunks.append(chunk)

    return chunks