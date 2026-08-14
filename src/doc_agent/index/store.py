"""Stage 4 - persist and load the FAISS vector store."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import Chunk


def _index_dir(cfg: dict) -> Path:
    path = cfg.get("index", {}).get("path", "data/index")
    return Path(path)


def _faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("The FAISS store requires faiss-cpu.") from exc
    return faiss


def build(chunks: list[Chunk], vectors: np.ndarray, cfg: dict) -> None:
    """Build and persist a normalized-vector FAISS HNSW index."""
    index_cfg = cfg.get("index", {})
    index_type = str(index_cfg.get("type", "faiss:hnsw")).lower()
    if index_type != "faiss:hnsw":
        raise ValueError(f"Unsupported index type: {index_type!r}")

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be a 2D array, got shape {vectors.shape}")
    if len(chunks) != vectors.shape[0]:
        raise ValueError(
            f"Chunk/vector count mismatch: {len(chunks)} chunks and {vectors.shape[0]} vectors"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("vectors contain NaN or infinite values")

    faiss = _faiss()
    dimension = vectors.shape[1]
    graph_degree = int(index_cfg.get("hnsw_m", 32))
    if graph_degree <= 0:
        raise ValueError("index.hnsw_m must be positive")

    index = faiss.IndexHNSWFlat(dimension, graph_degree, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = int(index_cfg.get("ef_construction", 40))
    index.hnsw.efSearch = int(index_cfg.get("ef_search", 64))
    index.add(vectors)

    output_dir = _index_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "index.faiss"))

    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")

    metadata = {
        "index_type": index_type,
        "dimension": dimension,
        "count": len(chunks),
        "metric": "inner_product",
        "hnsw_m": graph_degree,
        "ef_construction": index.hnsw.efConstruction,
        "ef_search": index.hnsw.efSearch,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load(cfg: dict) -> tuple[Any, list[Chunk]]:
    """Load the persisted FAISS index and its chunk metadata."""
    faiss = _faiss()
    output_dir = _index_dir(cfg)
    index_path = output_dir / "index.faiss"
    chunks_path = output_dir / "chunks.jsonl"
    metadata_path = output_dir / "metadata.json"
    missing = [str(path) for path in (index_path, chunks_path, metadata_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing index artifacts: " + ", ".join(missing))

    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata["dimension"]) != index.d:
        raise ValueError("Index dimension does not match metadata")
    if int(metadata["count"]) != index.ntotal:
        raise ValueError("Index vector count does not match metadata")

    chunks = [
        Chunk.model_validate(json.loads(line))
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(chunks) != index.ntotal:
        raise ValueError("Index vector count does not match stored chunks")
    return index, chunks
