"""Data — corpus versioning (which corpus version -> which result)"""

from __future__ import annotations

from ..contracts import *  # noqa


def snapshot(corpus_dir: str) -> str:
    """Hash + record a corpus version id. IMPLEMENT (or wire DVC)."""
    from hashlib import sha256
    from pathlib import Path

    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {root}")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Corpus directory contains no files: {root}")

    digest = sha256()
    digest.update(b"doc-agent-corpus-v1\0")
    for path in files:
        relative_path = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, byteorder="big"))
        digest.update(relative_path)
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)

    return f"sha256:{digest.hexdigest()}"
