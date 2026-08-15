"""Tests for corpus loading and page preprocessing."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from doc_agent.contracts import Page
from doc_agent.ingest import enhance, loader, preprocess


def _write_image(path: Path, size: tuple[int, int] = (160, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(230, 215, 190)).save(path)


def _preprocess_config(output_dir: Path, **overrides: object) -> dict:
    settings: dict[str, object] = {
        "enabled": True,
        "output_dir": str(output_dir),
        "trim_left_px": 80,
        "trim_border_px": 0,
        "deskew": False,
        "min_skew_degrees": 0.15,
        "max_skew_degrees": 2.0,
    }
    settings.update(overrides)
    return {"preprocess": settings}


def test_loader_filters_to_configured_chapter_pages(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_image(raw_dir / "krishipath" / "page-0001.png")
    _write_image(raw_dir / "krishipath" / "page-0002.jpg")
    _write_image(raw_dir / "krishipath" / "page-0003.png")
    _write_image(raw_dir / "krishipath" / "cover.png")

    pages = loader.load_pages(
        {
            "data": {
                "raw_dir": str(raw_dir),
                "chapter_pages_only": True,
                "chapter_ranges": {1: [1, 2]},
            }
        }
    )

    assert [page.id for page in pages] == ["krishipath/page-0001", "krishipath/page-0002"]
    assert {page.doc_id for page in pages} == {"krishipath"}


def test_loader_rejects_a_chapter_filter_with_no_matching_images(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_image(raw_dir / "page-0003.png")

    with pytest.raises(ValueError, match="No supported page images"):
        loader.load_pages(
            {
                "data": {
                    "raw_dir": str(raw_dir),
                    "chapter_pages_only": True,
                    "chapter_ranges": {1: [1, 2]},
                }
            }
        )


def test_preprocess_trims_the_left_gutter_without_changing_raw_image(tmp_path: Path) -> None:
    source_path = tmp_path / "raw" / "page-0001.png"
    _write_image(source_path)
    original_size = Image.open(source_path).size
    page = Page(id="page-0001", image_path=str(source_path), doc_id="krishipath")

    processed = preprocess.run([page], _preprocess_config(tmp_path / "interim"))

    assert Image.open(source_path).size == original_size
    assert len(processed) == 1
    assert processed[0].id == page.id
    assert processed[0].doc_id == page.doc_id
    assert Path(processed[0].image_path) == tmp_path / "interim" / "page-0001.png"
    assert Image.open(processed[0].image_path).size == (80, 80)


def test_preprocess_keeps_a_blank_page_when_no_skew_is_detected(tmp_path: Path) -> None:
    source_path = tmp_path / "raw" / "page-0001.png"
    _write_image(source_path, size=(200, 100))
    page = Page(id="page-0001", image_path=str(source_path), doc_id="krishipath")

    processed = preprocess.run([page], _preprocess_config(tmp_path / "interim", deskew=True))

    assert Image.open(processed[0].image_path).size == (120, 100)


def test_preprocess_disabled_returns_the_original_pages(tmp_path: Path) -> None:
    source_path = tmp_path / "raw" / "page-0001.png"
    _write_image(source_path)
    pages = [Page(id="page-0001", image_path=str(source_path), doc_id="krishipath")]

    processed = preprocess.run(pages, _preprocess_config(tmp_path / "interim", enabled=False))

    assert processed is pages
    assert not (tmp_path / "interim").exists()


def test_disabled_enhancement_returns_the_original_pages(tmp_path: Path) -> None:
    source_path = tmp_path / "raw" / "page-0001.png"
    _write_image(source_path)
    pages = [Page(id="page-0001", image_path=str(source_path), doc_id="krishipath")]

    enhanced = enhance.run(pages, {"enhance": {"enabled": False}})

    assert enhanced is pages
