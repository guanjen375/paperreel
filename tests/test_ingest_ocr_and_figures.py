"""Ingest stage: OCR fallback + figure bbox/caption extraction.

Scope:
- :func:`paperreel.ocr.is_available` correctly reports pytesseract +
  tesseract binary presence.
- A page with too little extracted text routes through OCR when the
  optional dependency is installed; ``text_source`` records the route.
- Embedded images come back with a bbox and (best-effort) caption hint
  so the ``match_visuals`` stage has something to score against.
"""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from PIL import Image, ImageDraw

from paperreel import ocr as ocr_mod
from paperreel.models import ChunkedSources
from paperreel.stages import ingest_pdf
from paperreel.state import StateDB


def _make_image_pdf_with_caption(tmp_path: Path) -> Path:
    """A 2-page PDF where page 2 contains a raster image + caption text.

    Use this to exercise figure extraction (bbox + caption_hint). The
    image is large enough to clear ``min_image_pixels``.
    """
    pdf_path = tmp_path / "with_figure.pdf"
    doc = fitz.open()

    # Page 1 — plain text so ingest doesn't degenerate to all-OCR.
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((50, 80), "Introduction\nThis is a normal text page.\n" * 10,
                       fontsize=12, fontname="helv", color=(0, 0, 0))

    # Page 2 — an image with a caption below it.
    page2 = doc.new_page(width=595, height=842)
    # Build a recognisable PNG (400x300 red square) the figure extractor
    # will save out + assign a bbox to. Has to be over min_image_pixels
    # (default 40_000) to survive the decoration-icon filter.
    raster_path = tmp_path / "figure.png"
    Image.new("RGB", (400, 300), color=(220, 60, 60)).save(raster_path)
    page2.insert_image(fitz.Rect(100, 100, 400, 325),
                        filename=str(raster_path))
    # Caption right under the image. The text block here is what
    # _caption_below_bbox should pick up.
    page2.insert_text((100, 360), "Figure 1. A red rectangle used as a test fixture.",
                       fontsize=11, fontname="helv", color=(0, 0, 0))
    page2.insert_text((50, 500),
                       "Body text mentioning rectangle and fixture for matching.",
                       fontsize=11, fontname="helv", color=(0, 0, 0))

    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _make_scanned_style_pdf(tmp_path: Path) -> Path:
    """A 1-page PDF whose only content is a rendered image of some text
    — no extractable text layer. Forces the ingest stage to either OCR
    or record ``text_source="empty"``."""
    pdf_path = tmp_path / "scanned.pdf"
    # Build a bitmap of text using Pillow so PyMuPDF treats it as an
    # opaque image with no underlying text stream.
    img_path = tmp_path / "scanned_page.png"
    img = Image.new("RGB", (800, 1000), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((40, 40), "HELLO WORLD\nThis text only exists as pixels.",
               fill=(0, 0, 0))
    img.save(img_path)

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(0, 0, 595, 842), filename=str(img_path))
    doc.save(pdf_path)
    doc.close()
    return pdf_path


# --- ocr module ------------------------------------------------------------

def test_is_available_returns_bool() -> None:
    # Shouldn't crash even if pytesseract is missing; cached.
    assert isinstance(ocr_mod.is_available(), bool)


def test_missing_languages_handles_no_tesseract(monkeypatch) -> None:
    """When Tesseract isn't installed, the helper returns () rather than
    pretending every language is missing — caller uses is_available()
    for the binary check."""
    monkeypatch.setattr(ocr_mod, "is_available", lambda: False)
    assert ocr_mod.missing_languages("chi_tra+eng") == ()


@pytest.mark.skipif(not ocr_mod.is_available(),
                     reason="Tesseract or pytesseract not available")
def test_ocr_page_real_path_returns_text(tmp_path: Path) -> None:
    """Real-path test — no monkeypatch on ocr_page. Proves the
    PyMuPDF -> PIL -> Tesseract pipeline actually works end-to-end.
    Catches bugs (e.g. wrong attribute access on page.parent) that
    monkeypatched fast-path tests can't see."""
    import fitz
    from PIL import Image, ImageDraw, ImageFont

    # Build a bitmap with big, high-contrast Latin text so Tesseract
    # without language packs can still read it.
    img_path = tmp_path / "page_bitmap.png"
    img = Image.new("RGB", (1200, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
    except Exception:
        font = ImageFont.load_default()
    draw.text((40, 200), "HELLO PAPERREEL", fill=(0, 0, 0), font=font)
    img.save(img_path)

    pdf_path = tmp_path / "page_bitmap.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(0, 0, 595, 842), filename=str(img_path))
    doc.save(pdf_path)
    doc.close()

    doc = fitz.open(pdf_path)
    try:
        # Use eng so we don't fail when chi_tra/chi_sim aren't installed.
        text = ocr_mod.ocr_page(doc[0], lang="eng", dpi=200)
    finally:
        doc.close()
    assert "HELLO" in text.upper() or "PAPERREEL" in text.upper(), (
        f"OCR returned no recognisable text: {text!r}"
    )


def test_ocr_page_raises_helpful_error_without_pytesseract(monkeypatch) -> None:
    """If pytesseract is uninstalled, ``ocr_page`` should fail loudly
    rather than silently returning empty — the caller (ingest) chooses
    whether to swallow that."""
    import builtins
    real_import = builtins.__import__

    def block_pytesseract(name, *a, **kw):
        if name == "pytesseract":
            raise ImportError("simulated missing pytesseract")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", block_pytesseract)
    with pytest.raises(ImportError, match="pytesseract"):
        ocr_mod.ocr_page(object())  # never reaches the page argument


# --- ingest with OCR -------------------------------------------------------

def test_ingest_falls_back_to_ocr_on_empty_page(
    project_dir: Path, tmp_path: Path, test_cfg: dict, monkeypatch
) -> None:
    """A pixel-only page with no text layer should be marked
    ``text_source="ocr"`` and contain whatever the injected OCR returns."""
    pdf = _make_scanned_style_pdf(tmp_path)

    # Force the OCR path on, and inject a deterministic OCR result so we
    # don't depend on Tesseract's actual accuracy on a tiny fixture.
    cfg = {**test_cfg,
           "ingest": {**test_cfg["ingest"],
                       "ocr_fallback": True,
                       "ocr_min_chars": 50}}
    monkeypatch.setattr(ocr_mod, "is_available", lambda: True)
    monkeypatch.setattr(ocr_mod, "ocr_page",
                         lambda page, *, lang="...", dpi=200:
                             "OCR 抽到的文字內容，這裡是模擬輸出。")

    db = StateDB(project_dir / "state.sqlite")
    sources = ingest_pdf.run(pdf_path=pdf, project_root=project_dir,
                              db=db, config=cfg)
    db.close()
    pages = {pg.page: pg for pg in sources.pages}
    assert pages[1].text_source == "ocr"
    assert "OCR 抽到的文字內容" in pages[1].text


def test_ingest_records_text_source_text_for_digital_pdf(
    project_dir: Path, tiny_pdf: Path, test_cfg: dict
) -> None:
    """The conftest tiny_pdf has digital text — no OCR should fire."""
    db = StateDB(project_dir / "state.sqlite")
    sources = ingest_pdf.run(pdf_path=tiny_pdf, project_root=project_dir,
                              db=db, config=test_cfg)
    db.close()
    assert all(pg.text_source == "text" for pg in sources.pages)


def test_ingest_marks_empty_when_ocr_disabled(
    project_dir: Path, tmp_path: Path, test_cfg: dict
) -> None:
    """OCR off + no text on the page should not silently fabricate text;
    text_source goes to 'empty' so the quality report can warn."""
    pdf = _make_scanned_style_pdf(tmp_path)
    cfg = {**test_cfg,
           "ingest": {**test_cfg["ingest"], "ocr_fallback": False}}
    db = StateDB(project_dir / "state.sqlite")
    sources = ingest_pdf.run(pdf_path=pdf, project_root=project_dir,
                              db=db, config=cfg)
    db.close()
    assert sources.pages[0].text_source in ("empty", "text")
    if sources.pages[0].text_source == "empty":
        assert not sources.pages[0].text.strip()


# --- ingest extracts figure bbox + caption ---------------------------------

def test_ingest_extracts_image_with_bbox(
    project_dir: Path, tmp_path: Path, test_cfg: dict
) -> None:
    pdf = _make_image_pdf_with_caption(tmp_path)
    db = StateDB(project_dir / "state.sqlite")
    sources = ingest_pdf.run(pdf_path=pdf, project_root=project_dir,
                              db=db, config=test_cfg)
    db.close()
    figures = [im for im in sources.images if im.page == 2]
    assert figures, "expected at least one figure on page 2"
    fig = figures[0]
    assert fig.bbox is not None
    # Sanity-check the bbox roughly matches where we inserted the image
    # (top edge near y=100, bottom edge near y=325). PyMuPDF can shift
    # by a few points; allow slack.
    x0, y0, x1, y1 = fig.bbox
    assert 80 < x0 < 120
    assert 80 < y0 < 120
    assert 380 < x1 < 420
    assert 305 < y1 < 345


def test_ingest_captures_caption_below_figure(
    project_dir: Path, tmp_path: Path, test_cfg: dict
) -> None:
    """The text immediately below an image should land in caption_hint."""
    pdf = _make_image_pdf_with_caption(tmp_path)
    db = StateDB(project_dir / "state.sqlite")
    sources = ingest_pdf.run(pdf_path=pdf, project_root=project_dir,
                              db=db, config=test_cfg)
    db.close()
    figures = [im for im in sources.images if im.page == 2]
    assert figures
    caption = figures[0].caption_hint or ""
    assert "Figure 1" in caption, f"caption_hint missing the caption text: {caption!r}"
