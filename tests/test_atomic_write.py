from __future__ import annotations

from pathlib import Path

import pytest

from pdf2lesson.io_utils import (atomic_write, atomic_write_json,
                                 atomic_write_text, read_json)


def test_atomic_text_creates_file(tmp_path: Path) -> None:
    p = tmp_path / "out.txt"
    atomic_write_text(p, "hello")
    assert p.read_text(encoding="utf-8") == "hello"
    # No leftover .tmp.* sibling
    siblings = list(tmp_path.iterdir())
    assert all(".tmp." not in s.name for s in siblings)


def test_atomic_json_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "data.json"
    payload = {"a": 1, "中文": ["甲", "乙"], "nested": {"x": True}}
    atomic_write_json(p, payload)
    assert read_json(p) == payload


def test_atomic_write_preserves_old_on_failure(tmp_path: Path) -> None:
    p = tmp_path / "data.txt"
    p.write_text("OLD", encoding="utf-8")
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_write(p, "w") as f:
            f.write("NEW-partial")
            raise RuntimeError("boom")
    # File still has old content; no .tmp.* leftovers.
    assert p.read_text(encoding="utf-8") == "OLD"
    leftover = [s for s in tmp_path.iterdir() if ".tmp." in s.name]
    assert leftover == []


def test_atomic_write_overwrites_existing(tmp_path: Path) -> None:
    p = tmp_path / "data.txt"
    p.write_text("OLD", encoding="utf-8")
    atomic_write_text(p, "NEW")
    assert p.read_text(encoding="utf-8") == "NEW"
