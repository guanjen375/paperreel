"""Stage 1 — ingest PDF -> chunked_sources.json (+ images dir).

Uses PyMuPDF (fitz). For each page we capture:

- Body text via :meth:`Page.get_text("text")`. When that yields fewer
  characters than ``ingest.ocr_min_chars`` and ``ingest.ocr_fallback``
  is enabled, we OCR the rendered page bitmap so scanned / image-only
  PDFs aren't silently dropped.
- Embedded raster images with their on-page bounding boxes and a best-
  effort caption hint sourced from the text immediately below each
  image. The ``match_pdf_visuals`` stage uses both to decide which
  figure to assign to which scene.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from .. import ocr as ocr_mod
from ..hashing import sha256_bytes, sha256_file, hash_inputs
from ..io_utils import atomic_write_bytes, atomic_write_json, ensure_dir
from ..models import (ChunkedSources, PdfChunk, PdfImage, PdfPage)
from ..state import StateDB
from ..utils.text_cleaning import cjk_char_count, extract_headings, normalise_text


@dataclass
class IngestPaths:
    root: Path
    chunked_json: Path
    images_dir: Path


def paths_for(project_root: str | Path) -> IngestPaths:
    root = Path(project_root)
    return IngestPaths(
        root=root,
        chunked_json=root / "intermediate" / "chunked_sources.json",
        images_dir=root / "assets" / "pdf_images",
    )


def _chunk_pages(pages: list[PdfPage], *, max_chars: int) -> list[PdfChunk]:
    chunks: list[PdfChunk] = []
    buf_text: list[str] = []
    buf_pages: list[int] = []
    buf_headings: list[str] = []
    buf_chars = 0
    for p in pages:
        if buf_chars + p.cjk_char_count > max_chars and buf_pages:
            chunks.append(_emit_chunk(buf_pages, buf_text, buf_headings, len(chunks)))
            buf_text, buf_pages, buf_headings, buf_chars = [], [], [], 0
        buf_text.append(p.text)
        buf_pages.append(p.page)
        buf_headings.extend(p.headings)
        buf_chars += p.cjk_char_count
    if buf_pages:
        chunks.append(_emit_chunk(buf_pages, buf_text, buf_headings, len(chunks)))
    return chunks


def _emit_chunk(pages: list[int], texts: list[str],
                headings: list[str], idx: int) -> PdfChunk:
    joined = "\n\n".join(texts).strip()
    return PdfChunk(
        chunk_id=f"chunk_{idx + 1:04d}",
        start_page=pages[0],
        end_page=pages[-1],
        text=joined,
        cjk_char_count=cjk_char_count(joined),
        headings=headings[:20],
    )


def _resolve_image_bbox(page: Any, xref: int) -> tuple[float, ...] | None:
    """Return the first on-page bbox for this image xref, or None.

    PyMuPDF can place the same xref at multiple rects (logos, headers).
    For caption-finding we take the first rect — multi-placement images
    rarely have meaningful captions anyway.
    """
    try:
        rects = page.get_image_rects(xref)
    except Exception:
        return None
    if not rects:
        return None
    r = rects[0]
    return (float(r.x0), float(r.y0), float(r.x1), float(r.y1))


def _caption_below_bbox(page: Any, bbox: tuple[float, ...],
                         *, max_gap: float = 80.0,
                         max_chars: int = 200) -> str | None:
    """Pick the text line(s) directly under ``bbox`` as a caption hint.

    Heuristic: look at text blocks whose y0 is within ``max_gap`` points
    below the image, and whose horizontal extent overlaps the image.
    Falls back to None when nothing looks captiony — better silent than
    fabricating a wrong caption.
    """
    try:
        blocks = page.get_text("blocks") or []
    except Exception:
        return None
    x0, _y0, x1, y1 = bbox
    width = max(1.0, x1 - x0)
    candidates: list[tuple[float, str]] = []
    for b in blocks:
        if len(b) < 5:
            continue
        bx0, by0, bx1, by1, btext = b[0], b[1], b[2], b[3], b[4]
        if not isinstance(btext, str):
            continue
        if by0 < y1 or (by0 - y1) > max_gap:
            continue
        overlap = max(0.0, min(bx1, x1) - max(bx0, x0))
        if overlap / width < 0.25:
            continue
        candidates.append((by0, btext.strip()))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    joined = " ".join(t for _, t in candidates if t)
    if not joined:
        return None
    return joined[:max_chars]


def _extract_page_images(
    page: Any, doc: Any, page_idx: int, *,
    images_dir: Path, min_pixels: int, img_fmt: str,
) -> list[PdfImage]:
    """Save embedded images for one page; return a list of PdfImage."""
    results: list[PdfImage] = []
    seen_xrefs: set[int] = set()
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            pix = fitz.Pixmap(doc, xref)
        except Exception:
            continue
        try:
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            pixels = pix.width * pix.height
            if pixels < min_pixels:
                continue
            ext = "png" if img_fmt == "png" else "jpg"
            data = pix.tobytes(ext)
            sha = sha256_bytes(data)
            img_path = images_dir / f"p{page_idx + 1:04d}_x{xref}_{sha[:8]}.{ext}"
            atomic_write_bytes(img_path, data)
            bbox = _resolve_image_bbox(page, xref)
            caption = _caption_below_bbox(page, bbox) if bbox else None
            results.append(PdfImage(
                image_id=f"img_p{page_idx + 1}_{xref}",
                page=page_idx + 1,
                path=str(img_path),
                width=pix.width,
                height=pix.height,
                pixel_count=pixels,
                sha256=sha,
                bbox=bbox,
                caption_hint=caption,
            ))
        finally:
            pix = None  # release native handle promptly
    return results


def _page_text_with_ocr(
    page: Any, *,
    ocr_enabled: bool,
    ocr_min_chars: int,
    ocr_lang: str,
    ocr_dpi: int,
) -> tuple[str, str]:
    """Return (text, source) — source is "text" / "ocr" / "empty".

    OCR is only triggered when the extracted text is short *and* the
    optional dependency is installed. Failing silently keeps offline
    environments working; the quality report flags any "empty" pages
    so the user notices before a full pipeline run wastes time.
    """
    raw_text = page.get_text("text") or ""
    text = normalise_text(raw_text)
    if len(text) >= max(1, ocr_min_chars):
        return text, "text"
    if not ocr_enabled or not ocr_mod.is_available():
        return text, ("text" if text else "empty")
    try:
        ocr_text = normalise_text(
            ocr_mod.ocr_page(page, lang=ocr_lang, dpi=ocr_dpi)
        )
    except Exception:
        return text, ("text" if text else "empty")
    if not ocr_text:
        return text, ("text" if text else "empty")
    # OCR result wins iff it actually beat the digital extraction.
    if len(ocr_text) > len(text):
        return ocr_text, "ocr"
    return text, ("text" if text else "empty")


def run(
    *,
    pdf_path: str | Path,
    project_root: str | Path,
    db: StateDB,
    config: dict,
    force: bool = False,
) -> ChunkedSources:
    paths = paths_for(project_root)
    ensure_dir(paths.chunked_json.parent)
    ensure_dir(paths.images_dir)

    ingest_cfg = config.get("ingest", {})
    llm_cfg = config.get("llm", {})

    ocr_enabled = bool(ingest_cfg.get("ocr_fallback", False))
    ocr_min_chars = int(ingest_cfg.get("ocr_min_chars", 50))
    ocr_lang = str(ingest_cfg.get("ocr_lang", "chi_tra+chi_sim+eng"))
    ocr_dpi = int(ingest_cfg.get("ocr_dpi", 200))

    pdf_sha = sha256_file(pdf_path)
    # Bump to v2 so existing projects pick up the new schema (text_source,
    # bbox, caption_hint on PdfImage) instead of reusing a stale
    # chunked_sources.json that pre-dates these fields.
    input_hash = hash_inputs(
        "ingest_v2", pdf_sha, ingest_cfg,
        llm_cfg.get("max_chunk_chars"),
        ocr_enabled, ocr_min_chars, ocr_lang, ocr_dpi,
        ocr_mod.is_available() if ocr_enabled else False,
    )
    outputs = [str(paths.chunked_json)]

    if not force and db.stage_is_done("ingest", input_hash, outputs):
        return ChunkedSources.model_validate_json(paths.chunked_json.read_text(encoding="utf-8"))

    db.start_stage("ingest", input_hash)
    try:
        doc = fitz.open(pdf_path)
        try:
            pages: list[PdfPage] = []
            images: list[PdfImage] = []
            heading_count = 0
            min_pixels = int(ingest_cfg.get("min_image_pixels", 40000))
            img_fmt = str(ingest_cfg.get("image_format", "png")).lower()
            extract_images = bool(ingest_cfg.get("extract_images", True))

            for pno in range(doc.page_count):
                page = doc.load_page(pno)
                text, source = _page_text_with_ocr(
                    page,
                    ocr_enabled=ocr_enabled,
                    ocr_min_chars=ocr_min_chars,
                    ocr_lang=ocr_lang,
                    ocr_dpi=ocr_dpi,
                )
                headings = extract_headings(text, max_lines=80)
                heading_count += len(headings)
                pages.append(PdfPage(
                    page=pno + 1,
                    text=text,
                    cjk_char_count=cjk_char_count(text),
                    headings=headings,
                    text_source=source,
                ))
                if not extract_images:
                    continue
                images.extend(_extract_page_images(
                    page, doc, pno,
                    images_dir=paths.images_dir,
                    min_pixels=min_pixels, img_fmt=img_fmt,
                ))

            total_cjk = sum(p.cjk_char_count for p in pages)
            chunks = _chunk_pages(pages, max_chars=int(llm_cfg.get("max_chunk_chars", 8000)))

            sources = ChunkedSources(
                source_pdf=str(pdf_path),
                pdf_sha256=pdf_sha,
                page_count=len(pages),
                cjk_char_count=total_cjk,
                image_count=len(images),
                heading_count=heading_count,
                estimated_density=(total_cjk / max(1, len(pages))),
                pages=pages,
                chunks=chunks,
                images=images,
            )
            atomic_write_json(paths.chunked_json, sources.model_dump(mode="json"))
            db.register_artifact(paths.chunked_json, stage="ingest",
                                 media_type="application/json",
                                 provenance={"source_pdf": str(pdf_path),
                                             "pdf_sha256": pdf_sha,
                                             "ocr_enabled": ocr_enabled,
                                             "ocr_available":
                                                 ocr_mod.is_available()
                                                 if ocr_enabled else False})
            for im in images:
                db.register_artifact(im.path, stage="ingest",
                                     media_type=f"image/{img_fmt}",
                                     provenance={"source_pdf": str(pdf_path),
                                                 "page": im.page,
                                                 "pdf_sha256": pdf_sha,
                                                 "bbox": im.bbox,
                                                 "caption_hint": im.caption_hint},
                                     compute_sha=False)
        finally:
            doc.close()
        db.finish_stage("ingest", outputs)
        return sources
    except Exception as e:
        db.fail_stage("ingest", repr(e))
        db.log_error("ingest", str(e))
        raise
