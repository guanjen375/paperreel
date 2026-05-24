"""Optional OCR fallback for image-only / scanned PDFs.

PyMuPDF's :meth:`Page.get_text("text")` returns an empty string for
pages that contain no embedded text — typical for scanned books,
projector slides exported as images, and photographed worksheets.
Those pages used to disappear silently from the lesson and surface
much later as a vacuous chapter ("(p.12 沒有可用內容)").

This module shells out to a local Tesseract install via ``pytesseract``
when the page text is below ``ingest.ocr_min_chars``. Both the Python
binding and the Tesseract binary are *optional*: if either is missing
:func:`is_available` returns False and the ingest stage just records
``text_source="empty"`` for the page so downstream stages can choose
their own behaviour (and the quality report can flag it).

Why no hard dependency: most projects targeting digital PDFs never need
OCR, and shipping Tesseract on every install would double the install
footprint. ``pip install paperreel[ocr]`` plus a system ``tesseract``
unlocks it.
"""
from __future__ import annotations

import shutil
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def is_available() -> bool:
    """True when both ``pytesseract`` AND the ``tesseract`` binary can
    be loaded. Cached because the answer doesn't change within a run."""
    try:
        import pytesseract  # type: ignore  # noqa: F401
    except ImportError:
        return False
    if shutil.which("tesseract") is None:
        # The python binding can import without the binary on PATH; that
        # combination still fails the first ``image_to_string`` call.
        return False
    return True


@lru_cache(maxsize=8)
def installed_languages() -> tuple[str, ...]:
    """Languages Tesseract can use. Empty tuple if Tesseract isn't
    installed. Cached because querying tesseract is a subprocess."""
    if not is_available():
        return ()
    try:
        import pytesseract  # type: ignore
        return tuple(pytesseract.get_languages(config=""))
    except Exception:
        return ()


def missing_languages(required: str) -> tuple[str, ...]:
    """Return the language packs from a ``+``-joined spec (e.g.
    ``chi_tra+eng``) that aren't installed locally. Empty tuple when
    everything's available or Tesseract isn't even installed (the
    latter case is caught by :func:`is_available` upstream)."""
    if not is_available():
        return ()
    want = [p.strip() for p in required.split("+") if p.strip()]
    have = set(installed_languages())
    return tuple(p for p in want if p not in have)


def ocr_page(page: Any, *, lang: str = "chi_tra+chi_sim+eng",
             dpi: int = 200) -> str:
    """OCR one PyMuPDF page and return the recognised text.

    Renders the page to a bitmap at ``dpi`` before handing it to
    Tesseract. Higher DPI improves accuracy on small fonts at the cost
    of seconds per page; 200 dpi is the sweet spot for typical 11pt
    book/paper layouts.

    Raises ImportError when the optional dependency isn't installed —
    the caller should gate the call with :func:`is_available`. Any
    Tesseract-level error (missing lang pack, corrupt page bitmap)
    is swallowed and returns an empty string so the ingest stage can
    fall back to ``text_source="empty"``.
    """
    try:
        import pytesseract  # type: ignore
        from PIL import Image
        # Imported here (not at module top) so users without the OCR
        # extra never pay the cost of loading fitz transitively.
        import fitz  # type: ignore
    except ImportError as e:
        raise ImportError(
            "OCR fallback requires pytesseract — install with "
            "`pip install paperreel[ocr]` and ensure the tesseract binary "
            "is on PATH."
        ) from e
    import io

    # Render the page at the requested dpi (72 dpi = native PDF units).
    zoom = dpi / 72.0
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    except Exception:
        # A corrupt page or a PyMuPDF render failure shouldn't poison
        # the whole pipeline — let ingest mark this page empty.
        return ""
    try:
        return pytesseract.image_to_string(img, lang=lang).strip()
    except pytesseract.TesseractError:
        # Bad lang pack / corrupt image -> let caller fall back to empty.
        return ""
    except Exception:
        return ""
