"""Tests for layout detection and OCR stage contracts."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from doc_agent.contracts import Page, Region
from doc_agent.vision import layout, ocr


class _FakeTensor:
    def __init__(self, values: list[Any]) -> None:
        self.values = values

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> list[Any]:
        return self.values


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, image_path: str, **kwargs: Any) -> list[Any]:
        self.calls.append({"image_path": image_path, **kwargs})
        boxes = SimpleNamespace(
            xyxy=_FakeTensor(
                [[15.0, 60.0, 70.0, 90.0], [100.0, 40.0, 220.0, 80.0], [20.0, 10.0, 80.0, 30.0]]
            ),
            cls=_FakeTensor([2.0, 1.0, 0.0]),
        )
        return [
            SimpleNamespace(
                orig_shape=(100, 200),
                boxes=boxes,
                names={0: "plain text", 1: "section_header", 2: "table"},
            )
        ]


class _FakeReader:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def read_page(self, image_path: str) -> list[dict[str, Any]]:
        del image_path
        return [
            {"bbox": (10.0, 10.0, 20.0, 20.0), "text": "প্রথম লাইন"},
            {"bbox": (10.0, 30.0, 20.0, 40.0), "text": "second line"},
        ]


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 100), color="white").save(path)


def _cuda_available(monkeypatch, available: bool) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available)),
    )


def test_device_selection_uses_cpu_when_cuda_is_unavailable(monkeypatch) -> None:
    _cuda_available(monkeypatch, available=False)

    assert layout._device("auto") == "cpu"
    assert layout._device("cuda:0") == "cpu"
    assert ocr._use_gpu("auto") is False
    assert ocr._use_gpu("cuda:0") is False


def test_device_selection_uses_cuda_when_available(monkeypatch) -> None:
    _cuda_available(monkeypatch, available=True)

    assert layout._device("auto") == "cuda:0"
    assert layout._device("cpu") == "cpu"
    assert ocr._use_gpu("auto") is True
    assert ocr._use_gpu("cpu") is False


def test_layout_detects_clips_and_orders_regions(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "page-0001.png"
    _write_image(image_path)
    model = _FakeModel()
    monkeypatch.setattr(layout, "_model", lambda _: model)
    page = Page(id="page-0001", image_path=str(image_path), doc_id="krishipath")
    cfg: dict[str, Any] = {
        "layout": {"device": "cpu", "score_thr": 0.3, "imgsz": 640},
    }

    regions = layout.detect([page], cfg)

    assert [region.kind for region in regions] == ["text", "heading", "table"]
    assert [region.bbox for region in regions] == [
        (20, 10, 80, 30),
        (100, 40, 200, 80),
        (15, 60, 70, 90),
    ]
    assert model.calls == [
        {
            "image_path": str(image_path),
            "imgsz": 640,
            "conf": 0.3,
            "device": "cpu",
            "verbose": False,
        }
    ]
    assert cfg["_page_map"] == {
        "page-0001": {"doc_id": "krishipath", "image_path": str(image_path)}
    }


def test_ocr_transcribes_each_source_page_once(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "page-0001.png"
    _write_image(image_path)
    monkeypatch.setattr(ocr, "Reader", _FakeReader)
    cfg = {
        "_page_map": {
            "page-0001": {"doc_id": "krishipath", "image_path": str(image_path)},
        }
    }
    regions = [
        Region(page_id="page-0001", bbox=(0, 0, 50, 50), kind="text"),
        Region(page_id="page-0001", bbox=(0, 50, 50, 100), kind="text"),
    ]

    chunks = ocr.transcribe(regions, cfg)

    assert len(chunks) == 1
    assert chunks[0].id == "page-0001_ocr"
    assert chunks[0].doc_id == "krishipath"
    assert chunks[0].page_ids == ["page-0001"]
    assert chunks[0].text == "প্রথম লাইন second line"


def test_ocr_returns_no_chunks_without_regions() -> None:
    assert ocr.transcribe([], {}) == []


def test_text_normalization_collapses_whitespace_and_uses_nfc() -> None:
    assert ocr._normalise_text("  ফসল\n\tরোগ  ") == "ফসল রোগ"
