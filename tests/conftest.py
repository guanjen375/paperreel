"""Shared pytest fixtures: minimal in-memory PDF + provider monkey-patching.

The production build only knows three real providers (ollama / xtts / sdxl),
all of which need GPU + downloaded weights. Tests cannot rely on any of
that, so an autouse fixture swaps the three factory functions for fakes
defined in `tests/_fakes/`. Production code is never modified.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

# Ensure src/ + tests/ are both importable without needing `pip install -e .`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT / "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)

import pytest

from paperreel.config import load_config
from paperreel.providers import image_base as _image_base
from paperreel.providers import llm_base as _llm_base
from paperreel.providers import tts_base as _tts_base

from _fakes.image_fake import FakeImage
from _fakes.llm_fake import FakeLLM
from _fakes.tts_fake import FakeTTS


@pytest.fixture(autouse=True)
def _patch_local_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap real local backends for deterministic fakes in every test.

    Tests should NEVER reach Ollama / XTTS / SDXL — they need GPU + weights
    that we don't ship to CI."""
    monkeypatch.setattr(_llm_base, "make_llm_provider", lambda cfg: FakeLLM(cfg))
    monkeypatch.setattr(_tts_base, "make_tts_provider", lambda cfg: FakeTTS(cfg))
    monkeypatch.setattr(_image_base, "make_image_provider", lambda cfg: FakeImage(cfg))
    # Stages import the factories by name (`from .llm_base import make_llm_provider`),
    # so we also have to patch the bound names inside each stage module.
    import paperreel.stages.build_outline as _bo
    import paperreel.stages.render_visuals as _rv
    import paperreel.stages.synthesize_audio as _sa
    import paperreel.stages.write_script as _ws
    monkeypatch.setattr(_bo, "make_llm_provider", lambda cfg: FakeLLM(cfg))
    monkeypatch.setattr(_ws, "make_llm_provider", lambda cfg: FakeLLM(cfg))
    monkeypatch.setattr(_sa, "make_tts_provider", lambda cfg: FakeTTS(cfg))
    monkeypatch.setattr(_rv, "make_image_provider", lambda cfg: FakeImage(cfg))


@pytest.fixture
def test_cfg() -> dict:
    """default.yaml with overrides that keep tests fast: tiny resolution,
    serial execution, and short budgets. Provider names are irrelevant
    because the autouse fixture above replaces them with fakes."""
    cfg = load_config()
    overrides = {
        "project": {"style": "default"},
        "tts": {"sample_rate_hz": 24000},
        "renderer": {"resolution": [1280, 720], "fps": 24},
        "runtime": {"max_hours": 0.5, "parallelism": 1},
    }
    return _deep_merge(cfg, overrides)


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@pytest.fixture
def tiny_pdf(tmp_path: Path) -> Path:
    """Create a 4-page CJK PDF in tmp_path/tiny.pdf."""
    import fitz  # PyMuPDF

    pdf_path = tmp_path / "tiny.pdf"
    doc = fitz.open()
    pages = [
        "第一章 緒論\n本書的目標是介紹一個小型範例。我們會在每個章節說明重點觀念。" * 4,
        "第二章 方法\n我們先用簡單的範例介紹整體流程。請注意每一步驟的目的與限制。" * 4,
        "1.1 入門\n這一節介紹基礎名詞與概念。讀者只需具備基本程式設計經驗。" * 4,
        "結語\n本章回顧全書重點。請務必在下一章開始前完成練習。" * 4,
    ]
    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_text(
            (50, 80), text,
            fontsize=12,
            fontname="china-s",
            color=(0, 0, 0),
        )
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    p = tmp_path / "project"
    p.mkdir()
    return p
