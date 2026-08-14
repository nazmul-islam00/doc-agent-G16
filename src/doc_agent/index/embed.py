"""Stage 4 - encode chunks as normalized dense vectors."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import Chunk

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _device(cfg: dict) -> str:
    """Select a usable torch device without requiring CUDA on every machine."""
    requested = str(cfg.get("embed", {}).get("device", cfg.get("device", "cpu")))
    if requested.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return requested
        except ImportError:
            pass
        return "cpu"
    return requested


def _get_model(cfg: dict) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Embedding requires sentence-transformers. Install the project dependencies."
        ) from exc

    embed_cfg = cfg.get("embed", {})
    model_name = str(embed_cfg.get("model", "BAAI/bge-m3"))
    device = _device(cfg)
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[key]


def encode(chunks: list[Chunk], cfg: dict) -> np.ndarray:
    """Encode chunks with the configured embedding model.

    Embeddings are L2-normalized so inner-product FAISS indexes behave as
    cosine-similarity indexes. The returned array is always float32 with shape
    ``(number_of_chunks, configured_dimension)``.
    """
    embed_cfg = cfg.get("embed", {})
    expected_dim = int(embed_cfg.get("dim", 1024))
    if expected_dim <= 0:
        raise ValueError("embed.dim must be positive")

    if not chunks:
        return np.empty((0, expected_dim), dtype=np.float32)

    model = _get_model(cfg)
    actual_dim = model.get_sentence_embedding_dimension()
    if actual_dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: config requests {expected_dim}, "
            f"but the model produces {actual_dim}."
        )

    batch_size = int(embed_cfg.get("batch_size", 16))
    if batch_size <= 0:
        raise ValueError("embed.batch_size must be positive")

    texts = [chunk.text for chunk in chunks]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=bool(embed_cfg.get("show_progress", False)),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(chunks), expected_dim):
        raise ValueError(
            f"Unexpected embedding shape {vectors.shape}; "
            f"expected {(len(chunks), expected_dim)}."
        )
    return vectors
