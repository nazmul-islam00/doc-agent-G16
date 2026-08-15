"""Tests for corpus validation and deterministic version snapshots."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_agent.contracts import Page
from doc_agent.data import validate as validation
from doc_agent.data.versioning import snapshot


def _set_validation_config(
    monkeypatch: pytest.MonkeyPatch,
    minimum_pages: int = 1,
    chapter_ranges: dict[int, list[int]] | None = None,
    chapter_splits: dict[str, list[int]] | None = None,
) -> None:
    ranges = chapter_ranges or {1: [1, 2]}
    splits = chapter_splits or {"train": [1], "validation": [], "test": []}
    monkeypatch.setattr(
        validation.config,
        "load_task",
        lambda: {"corpus": {"min_pages": minimum_pages}},
    )
    monkeypatch.setattr(
        validation.config,
        "load",
        lambda: {
            "data": {
                "chapter_pages_only": True,
                "chapter_ranges": ranges,
                "chapter_splits": splits,
            }
        },
    )


def _page(path: Path, page_id: str = "page-0001") -> Page:
    path.touch()
    return Page(id=page_id, image_path=str(path), doc_id="krishipath")


def test_validate_accepts_valid_page_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_validation_config(monkeypatch)
    validation.validate([_page(tmp_path / "page-0001.png")])


def test_validate_enforces_configured_minimum_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_validation_config(monkeypatch, minimum_pages=2)
    with pytest.raises(ValueError, match="at least 2"):
        validation.validate([_page(tmp_path / "page-0001.png")])


def test_validate_rejects_duplicate_page_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_validation_config(monkeypatch)
    pages = [
        _page(tmp_path / "page-0001.png"),
        _page(tmp_path / "page-0002.png"),
    ]
    with pytest.raises(ValueError, match="Duplicate page ID"):
        validation.validate(pages)


def test_validate_rejects_missing_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_validation_config(monkeypatch)
    missing = Page(
        id="page-0001",
        image_path=str(tmp_path / "missing.png"),
        doc_id="krishipath",
    )
    with pytest.raises(FileNotFoundError, match="missing image"):
        validation.validate([missing])


def test_validate_rejects_duplicate_image_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_validation_config(monkeypatch)
    source = tmp_path / "page.png"
    pages = [
        _page(source, "page-0001"),
        Page(id="page-0002", image_path=str(source), doc_id="krishipath"),
    ]
    with pytest.raises(ValueError, match="Duplicate image path"):
        validation.validate(pages)


def test_validate_rejects_auxiliary_page_when_chapter_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_validation_config(monkeypatch, chapter_ranges={1: [1, 1]})
    with pytest.raises(ValueError, match="not in a configured chapter range"):
        validation.validate([_page(tmp_path / "page-0002.png", "page-0002")])


def test_validate_rejects_overlapping_chapter_split_assignments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_validation_config(
        monkeypatch,
        chapter_ranges={1: [1, 1], 2: [2, 2]},
        chapter_splits={"train": [1], "validation": [1], "test": [2]},
    )
    with pytest.raises(ValueError, match="both 'train' and 'validation' splits"):
        validation.validate([_page(tmp_path / "page-0001.png")])


def test_snapshot_is_stable_for_unchanged_corpus(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    first = corpus_dir / "chapter-01" / "page-0001.txt"
    second = corpus_dir / "chapter-02" / "page-0002.txt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"first page")
    second.write_bytes(b"second page")

    first_hash = snapshot(str(corpus_dir))
    second_hash = snapshot(str(corpus_dir))

    assert first_hash == second_hash
    assert first_hash.startswith("sha256:")
    assert len(first_hash) == len("sha256:") + 64


def test_snapshot_changes_when_corpus_content_changes(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    page = corpus_dir / "page-0001.txt"
    page.write_bytes(b"original text")
    original_hash = snapshot(str(corpus_dir))

    page.write_bytes(b"revised text")

    assert snapshot(str(corpus_dir)) != original_hash


def test_snapshot_rejects_missing_or_empty_corpus(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        snapshot(str(tmp_path / "missing"))

    empty_corpus = tmp_path / "empty"
    empty_corpus.mkdir()
    with pytest.raises(ValueError, match="contains no files"):
        snapshot(str(empty_corpus))
