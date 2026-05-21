"""Generate examples/tiny_sample/tiny.pdf for manual / smoke testing."""
from __future__ import annotations

import pathlib

import fitz  # PyMuPDF


def main() -> None:
    out = pathlib.Path(__file__).resolve().parent / "tiny.pdf"
    doc = fitz.open()
    pages = [
        "第一章 緒論\n本書介紹簡短範例，用來測試整個 pipeline 的行為。"
        "我們會在每個章節說明重點觀念。" * 4,
        "第二章 方法\n說明流程與重點，使用者可從這裡了解步驟。" * 5,
        "1.1 入門範例\n這一節介紹基礎名詞與概念，並提供一個小範例。" * 4,
        "結語\n回顧所有重點。請務必在下一章開始前完成練習。" * 5,
    ]
    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_text(
            (50, 80), text,
            fontsize=12,
            fontname="china-s",
            color=(0, 0, 0),
        )
    doc.save(out)
    doc.close()
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
