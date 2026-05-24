"""PyMuPDF-backed crop / highlight helper.

Used by sketchbook ``source_crop`` scenes. We never paste an entire
shrunken PDF page into a card — instead, we find the bbox of the quoted
text on its source page, expand it slightly, and render only that
region as a PNG. The result is paired with a large explanation card in
the renderer (see ``SketchbookRenderer._source_crop``).

Crops are cached at ``assets/source_crops/<sha>.png`` keyed by
(pdf_sha256, page, bbox-or-quote, dpi, padding) so we don't re-render
identical inputs.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF

from ..hashing import hash_inputs
from ..io_utils import ensure_dir


@dataclass(frozen=True)
class CropResult:
    path: Path
    page: int
    bbox: tuple[float, float, float, float] | None
    quote: str
    readable: bool       # False when the crop is too small to be useful


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return "".join(s.split())


def _find_quote_bbox(page: fitz.Page, quote: str
                      ) -> tuple[float, float, float, float] | None:
    """Find the bounding box of ``quote`` on ``page``.

    PyMuPDF's ``search_for`` returns rects for exact matches. We try the
    full quote first, then progressively shorter prefixes until we get
    a hit. Returns the union rect (so multi-line quotes are bracketed
    together).
    """
    if not quote:
        return None
    candidates: list[str] = []
    norm = _normalize(quote)
    if quote:
        candidates.append(quote)
    if len(norm) > 8:
        candidates.append(quote[:max(12, len(quote) // 2)])
        candidates.append(quote[:max(8, len(quote) // 3)])
    for q in candidates:
        q = q.strip()
        if not q:
            continue
        try:
            rects = page.search_for(q)
        except Exception:
            continue
        if not rects:
            continue
        # Union of all matches so a quote that wraps lines becomes one
        # bbox instead of two disjoint slivers.
        x0 = min(r.x0 for r in rects)
        y0 = min(r.y0 for r in rects)
        x1 = max(r.x1 for r in rects)
        y1 = max(r.y1 for r in rects)
        return (x0, y0, x1, y1)
    return None


def _expand_bbox(bbox: tuple[float, float, float, float],
                 page_rect: fitz.Rect, *, padding_pt: float
                 ) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    x0 = max(page_rect.x0, x0 - padding_pt)
    y0 = max(page_rect.y0, y0 - padding_pt)
    x1 = min(page_rect.x1, x1 + padding_pt)
    y1 = min(page_rect.y1, y1 + padding_pt * 1.4)
    return (x0, y0, x1, y1)


def render_crop(*, pdf_path: str | Path, pdf_sha256: str,
                page: int, quote: str | None,
                bbox: tuple[float, float, float, float] | None = None,
                out_dir: Path,
                target_dpi: int = 220,
                padding_pt: float = 18.0,
                min_readable_height_pt: float = 80.0,
                highlight: bool = True,
                ) -> CropResult | None:
    """Render one crop. Returns ``None`` only when the page can't be
    opened — a successful crop returns a :class:`CropResult` with
    ``readable=False`` when the bbox is too small to bother showing.
    """
    ensure_dir(out_dir)
    # Cache by every input that affects the rendered pixels.
    key = hash_inputs(
        "pdf_crop_v1", pdf_sha256, page,
        bbox, quote or "", target_dpi, padding_pt, highlight,
    )
    out_path = out_dir / f"{key}.png"
    if out_path.exists():
        return CropResult(
            path=out_path,
            page=page,
            bbox=bbox,
            quote=quote or "",
            readable=True,
        )

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return None
    try:
        if page < 1 or page > doc.page_count:
            return None
        pg = doc.load_page(page - 1)
        rect = pg.rect
        resolved_bbox = bbox or _find_quote_bbox(pg, quote or "")
        if resolved_bbox is None:
            # Without a bbox we can't safely crop; refuse so the caller
            # falls back to a paragraph card. Better than rendering a
            # tiny full-page screenshot.
            return None
        expanded = _expand_bbox(resolved_bbox, rect, padding_pt=padding_pt)
        clip = fitz.Rect(*expanded)
        if (clip.y1 - clip.y0) < min_readable_height_pt:
            return CropResult(
                path=out_path, page=page,
                bbox=resolved_bbox, quote=quote or "",
                readable=False,
            )
        zoom = target_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        if highlight and resolved_bbox is not None:
            try:
                annot = pg.add_highlight_annot(fitz.Rect(*resolved_bbox))
                annot.set_colors(stroke=(1.0, 0.85, 0.2))
                annot.update()
            except Exception:
                pass
        pix = pg.get_pixmap(matrix=matrix, clip=clip, alpha=False)
        pix.save(str(out_path))
        return CropResult(
            path=out_path, page=page,
            bbox=resolved_bbox, quote=quote or "",
            readable=True,
        )
    finally:
        doc.close()


def best_crop_for_evidence(*, pdf_path: str | Path, pdf_sha256: str,
                           evidence_pages_and_quotes: Iterable[tuple[int, str]],
                           out_dir: Path,
                           target_dpi: int = 220,
                           padding_pt: float = 18.0,
                           min_readable_height_pt: float = 80.0,
                           ) -> CropResult | None:
    """Try each (page, quote) pair until one yields a readable crop."""
    for page, quote in evidence_pages_and_quotes:
        result = render_crop(
            pdf_path=pdf_path, pdf_sha256=pdf_sha256,
            page=page, quote=quote,
            out_dir=out_dir,
            target_dpi=target_dpi,
            padding_pt=padding_pt,
            min_readable_height_pt=min_readable_height_pt,
        )
        if result and result.readable:
            return result
    return None
