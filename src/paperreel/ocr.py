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


def ocr_page(page: Any, *, lang: str = "chi_tra+chi_sim+eng",
             dpi: int = 200) -> str:
    """OCR one PyMuPDF page and return the recognised text.

    Renders the page to a bitmap at ``dpi`` before handing it to
    Tesseract. Higher DPI improves accuracy on small fonts at the cost
    of seconds per page; 200 dpi is the sweet spot for typical 11pt
    book/paper layouts.

    Raises ImportError when the optional dependency isn't installed —
    the caller should gate the call with :func:`is_available`.
    """
    try:
        import pytesseract  # type: ignore
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "OCR fallback requires pytesseract — install with "
            "`pip install paperreel[ocr]` and ensure the tesseract binary "
            "is on PATH."
        ) from e
    import io

    # Tesseract works on a flat bitmap; rendering at the requested dpi
    # is the standard PyMuPDF workflow.
    zoom = dpi / 72.0
    matrix = page.parent.PDF_MATRIX_IDENTITY  # type: ignore[attr-defined]
    # Use Matrix(zoom, zoom) without importing fitz at module load.
    import fitz  # type: ignore  # local: keeps this module lazy for non-OCR users
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    try:
        return pytesseract.image_to_string(img, lang=lang).strip()
    except pytesseract.TesseractError:
        # Bad lang pack / corrupt image -> let caller fall back to empty.
        return ""
    except Exception:
        return ""
