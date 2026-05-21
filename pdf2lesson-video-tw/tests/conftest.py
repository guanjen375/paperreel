"""Shared pytest fixtures: minimal in-memory PDF + project scaffolding."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable without needing `pip install -e .`
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest


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
            # PyMuPDF needs a CJK-capable font; "china-s" is shipped as builtin
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
