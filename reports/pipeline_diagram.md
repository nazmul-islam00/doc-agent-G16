# Knowledge-base pipeline diagram (A2)

This diagram covers the implemented A2 knowledge-base build only. Retrieval,
the agent loop, reranking, citations, serving, and RL are deliberately omitted
because they are later-milestone work.

```mermaid
flowchart TD
    C[configs/config.yaml] --> B[build_knowledge_base]
    R[Raw Krishipath scans<br/>403 rendered pages] --> L[loader.load_pages]
    B --> W[wiring.register_all]
    W --> L
    L --> F[Filter configured chapter ranges<br/>378 selected pages]
    F --> P[preprocess.run<br/>EXIF transpose, 80 px left trim,<br/>bounded Hough-line deskew]
    P --> E[enhance.run<br/>disabled: pass-through]
    E --> H1[AFTER_INGEST hook]
    H1 --> D[layout.detect<br/>DocLayout-YOLO / YOLOv10]
    D --> M[Region records + page metadata map]
    M --> O[ocr.transcribe<br/>EasyOCR full page, bn + en]
    O --> H2[AFTER_OCR hook]
    H2 --> K[chunk.split<br/>NFC cleanup + BGE tokenizer<br/>256 tokens, 32-token overlap]
    K --> H3[BEFORE_INDEX hook]
    H3 --> V[embed.encode<br/>BAAI/bge-m3, 1024 dimensions,<br/>L2-normalized vectors]
    V --> S[store.build<br/>FAISS HNSW inner product]
    S --> A[index.faiss + chunks.jsonl + metadata.json]

    X[25 auxiliary scans] -. excluded by configured ranges .-> F
    G[57 held-out page labels] -. evaluation only; never indexed .-> Q[OCR CER / WER / character F1]
    O -. predicted OCR text .-> Q
```

## Stage behavior

| Stage | Implemented behavior |
|---|---|
| Load | `loader.load_pages()` discovers supported image files and keeps only pages in configured chapter ranges. |
| Preprocess | `preprocess.run()` writes derived PNGs without changing raw images. It applies EXIF orientation, a left-gutter trim, and bounded consensus deskewing. |
| Enhance | `enhance.run()` remains in the fixed pipeline order but is a pass-through because `enhance.enabled: false`. No optional enhancement model is part of the A2 run. |
| Layout | `layout.detect()` uses `juliozhao/DocLayout-YOLO-DocStructBench` through DocLayout-YOLO / YOLOv10. Regions provide structural metadata and reading order. |
| OCR | `ocr.transcribe()` uses EasyOCR with Bengali and English. It transcribes each processed page in full rather than only detected boxes, preventing observed crop-related omissions. |
| Chunk | `chunk.split()` NFC-normalizes source OCR text, preserves paragraph grouping, and creates BGE-tokenizer windows of 256 tokens with 32-token overlap. |
| Embed | `embed.encode()` creates 1024-dimensional, L2-normalized BGE-M3 vectors. |
| Store | `store.build()` writes a FAISS `IndexHNSWFlat` inner-product index plus JSONL chunks and JSON metadata. |

## Runtime and audit boundaries

The committed configuration uses `device: cpu` for a reliable baseline on the
teammate machine whose installed Torch build cannot run kernels on its `sm_120`
GPU. The standalone Kaggle preprocessing/OCR script accepts an explicit GPU
selection for a compatible Kaggle accelerator; that setting does not alter the
committed A2 configuration.

`data.validate.validate()` and `data.versioning.snapshot()` are available
integrity utilities, but `build_knowledge_base()` does not invoke them yet.
They should be run and their results recorded with the final index build rather
than represented as automatic pipeline artifacts.

## Produced artifacts

| Artifact | Location |
|---|---|
| Raw scans | `data/raw/krishipath/` |
| Derived scans | `data/interim/preprocessed/` |
| FAISS index | `data/index/index.faiss` |
| Chunk records | `data/index/chunks.jsonl` |
| Index metadata | `data/index/metadata.json` |
| Held-out OCR labels | `grading_kit/heldout_pages/` and `grading_kit/labels.jsonl` |
