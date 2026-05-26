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
from ..models import (ChunkedSources, PdfChunk, PdfImage, PdfPage,
                       VisualCandidate)
from ..state import StateDB
from ..utils.text_cleaning import cjk_char_count, extract_headings, normalise_text


@dataclass
class IngestPaths:
    root: Path
    chunked_json: Path
    images_dir: Path
    page_renders_dir: Path


def paths_for(project_root: str | Path) -> IngestPaths:
    root = Path(project_root)
    return IngestPaths(
        root=root,
        chunked_json=root / "intermediate" / "chunked_sources.json",
        images_dir=root / "assets" / "pdf_images",
        page_renders_dir=root / "assets" / "pdf_page_renders",
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


_TABLE_TERMS = ("表", "table", "欄", "列", "row", "column")
_SCREENSHOT_TERMS = ("截圖", "畫面", "介面", "選單", "按鈕", "menu", "button", "screenshot", "panel")
_CHART_TERMS = ("圖表", "直方圖", "histogram", "chart", "curve", "graph", "色階")
_DIAGRAM_TERMS = ("流程", "步驟", "示意", "diagram", "三角", "結構", "原理")
_PHOTO_TERMS = ("相片", "照片", "攝影", "範例", "實戰", "photo", "image", "example")
_DECORATIVE_TERMS = ("logo", "標誌", "版權", "copyright", "印章", "圖章", "seal", "stamp", "簽名", "signature")


def _safe_area_ratio(bbox: tuple[float, ...] | None,
                     page_width: float | None,
                     page_height: float | None) -> float:
    if not bbox or not page_width or not page_height:
        return 0.0
    x0, y0, x1, y1 = bbox
    page_area = max(1.0, float(page_width) * float(page_height))
    return max(0.0, min(1.0, ((x1 - x0) * (y1 - y0)) / page_area))


def _page_text_area_ratio(page: Any) -> float:
    try:
        blocks = page.get_text("blocks") or []
    except Exception:
        return 0.0
    page_area = max(1.0, float(page.rect.width) * float(page.rect.height))
    area = 0.0
    for b in blocks:
        if len(b) < 5 or not isinstance(b[4], str) or not b[4].strip():
            continue
        # PyMuPDF block type is index 6 when present; 0 means text.
        if len(b) >= 7 and b[6] != 0:
            continue
        area += max(0.0, float(b[2] - b[0]) * float(b[3] - b[1]))
    return round(min(1.0, area / page_area), 4)


def _page_image_area_ratio(page_images: list[PdfImage],
                           page_width: float, page_height: float) -> float:
    return round(min(1.0, sum(
        _safe_area_ratio(im.bbox, page_width, page_height)
        for im in page_images
    )), 4)


def _first_heading(page: PdfPage) -> str | None:
    for heading in page.headings:
        h = str(heading).strip()
        if h:
            return h[:80]
    for line in page.text.splitlines():
        s = line.strip()
        if 4 <= len(s) <= 80:
            return s[:80]
    return None


def _short_nearby_text(text: str, caption: str | None = None) -> str | None:
    clean = " ".join(s.strip() for s in text.splitlines() if s.strip())
    if not clean:
        return caption[:220] if caption else None
    if caption and caption in clean:
        idx = clean.find(caption)
        start = max(0, idx - 120)
        end = min(len(clean), idx + len(caption) + 160)
        return clean[start:end].strip()[:300]
    return clean[:300]


def _source_quote(text: str, heading: str | None, caption: str | None) -> str | None:
    for candidate in (caption, heading):
        if candidate and candidate in text:
            return candidate[:160]
    for line in text.splitlines():
        s = line.strip()
        if 8 <= len(s) <= 160:
            return s[:160]
    return None


def _looks_decorative(*, role: str, area_ratio: float, repeated: bool,
                      bbox: tuple[float, ...] | None,
                      page: PdfPage, blob: str) -> bool:
    if role in {"decorative", "logo", "seal", "signature"}:
        return True
    if any(term in blob for term in _DECORATIVE_TERMS) and area_ratio < 0.12:
        return True
    if repeated and area_ratio < 0.08:
        return True
    if bbox and page.width and page.height and area_ratio < 0.05:
        x0, y0, x1, y1 = bbox
        near_edge = y1 < page.height * 0.18 or y0 > page.height * 0.82
        small_edge = near_edge and (x1 - x0) < page.width * 0.35
        if small_edge:
            return True
    return False


def _classify_visual_role(*, caption: str | None, heading: str | None,
                          nearby_text: str | None, area_ratio: float,
                          repeated: bool, bbox: tuple[float, ...] | None,
                          page: PdfPage) -> str:
    blob = " ".join([caption or "", heading or "", nearby_text or ""]).lower()
    if "簽名" in blob or "signature" in blob:
        return "signature"
    if "印章" in blob or "圖章" in blob or "seal" in blob or "stamp" in blob:
        return "seal"
    if ("logo" in blob or "標誌" in blob) and area_ratio < 0.12:
        return "logo"
    if repeated and area_ratio < 0.06:
        return "decorative"
    if any(term.lower() in blob for term in _TABLE_TERMS):
        return "source_table"
    if any(term.lower() in blob for term in _SCREENSHOT_TERMS):
        return "source_screenshot"
    if any(term.lower() in blob for term in _CHART_TERMS):
        return "source_chart"
    if any(term.lower() in blob for term in _DIAGRAM_TERMS):
        return "source_diagram"
    if any(term.lower() in blob for term in _PHOTO_TERMS):
        return "source_photo"
    if area_ratio >= 0.16:
        return "source_photo"
    if bbox and page.width and page.height:
        x0, y0, x1, y1 = bbox
        if (x1 - x0) < page.width * 0.25 and (y1 - y0) < page.height * 0.18:
            return "decorative"
    return "unknown"


def _salience_score(*, role: str, area_ratio: float, pixel_count: int,
                    caption: str | None, heading: str | None,
                    repeated: bool, decorative: bool) -> float:
    score = 0.0
    score += min(4.0, area_ratio * 12.0)
    score += min(2.0, pixel_count / 500_000.0)
    if caption:
        score += 0.8
    if heading:
        score += 0.4
    if role in {"source_photo", "source_diagram", "source_table", "source_screenshot", "source_chart"}:
        score += 1.0
    if role == "source_page_crop":
        score += 0.5
    if repeated:
        score -= 0.8
    if decorative:
        score -= 3.0
    return round(max(0.0, score), 3)


def _page_visual_role(text: str) -> str:
    lower = text.lower()
    if any(t.lower() in lower for t in _TABLE_TERMS):
        return "source_table"
    if any(t.lower() in lower for t in _SCREENSHOT_TERMS):
        return "source_screenshot"
    if any(t.lower() in lower for t in _CHART_TERMS):
        return "source_chart"
    if any(t.lower() in lower for t in _DIAGRAM_TERMS):
        return "source_diagram"
    if any(t.lower() in lower for t in _PHOTO_TERMS):
        return "source_page_crop"
    return "source_page_crop"


def _render_page_png(doc: Any, page_no: int, renders_dir: Path,
                     *, dpi: int = 150) -> tuple[str, int, int] | None:
    ensure_dir(renders_dir)
    try:
        page = doc.load_page(page_no - 1)
        scale = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        data = pix.tobytes("png")
        out = renders_dir / f"p{page_no:04d}_page.png"
        atomic_write_bytes(out, data)
        return str(out), int(pix.width), int(pix.height)
    except Exception:
        return None


def _build_visual_inventory(*, pages: list[PdfPage], images: list[PdfImage],
                            doc: Any, renders_dir: Path) -> list[VisualCandidate]:
    pages_by_no = {p.page: p for p in pages}
    sha_counts: dict[str, int] = {}
    for im in images:
        sha_counts[im.sha256] = sha_counts.get(im.sha256, 0) + 1

    candidates: list[VisualCandidate] = []
    for im in images:
        page = pages_by_no.get(im.page)
        if page is None:
            continue
        area_ratio = _safe_area_ratio(im.bbox, page.width, page.height)
        heading = _first_heading(page)
        caption = im.caption_hint
        nearby = _short_nearby_text(page.text, caption)
        repeated = sha_counts.get(im.sha256, 0) > 2
        role = _classify_visual_role(
            caption=caption, heading=heading, nearby_text=nearby,
            area_ratio=area_ratio, repeated=repeated, bbox=im.bbox, page=page,
        )
        blob = " ".join([caption or "", heading or "", nearby or ""]).lower()
        decorative = _looks_decorative(
            role=role, area_ratio=area_ratio, repeated=repeated,
            bbox=im.bbox, page=page, blob=blob,
        )
        useful = (
            not decorative
            and role not in {"decorative", "logo", "seal", "signature"}
            and (area_ratio >= 0.045 or im.pixel_count >= 90_000 or bool(caption))
        )
        score = _salience_score(
            role=role, area_ratio=area_ratio, pixel_count=im.pixel_count,
            caption=caption, heading=heading, repeated=repeated,
            decorative=decorative,
        )
        candidates.append(VisualCandidate(
            candidate_id=f"vis_p{im.page:04d}_{im.image_id}",
            page=im.page,
            image_path=im.path,
            bbox=im.bbox,
            nearby_heading=heading,
            nearby_caption=caption,
            nearby_text=nearby,
            image_width=im.width,
            image_height=im.height,
            image_size=(im.width, im.height),
            page_area_ratio=round(area_ratio, 4),
            visual_role=role,
            salience_score=score,
            is_decorative=decorative,
            likely_useful=useful,
            repeated=repeated,
            source_image_id=im.image_id,
            source_quote=_source_quote(page.text, heading, caption),
        ))

    useful_pages = {c.page for c in candidates if c.likely_useful}
    page_render_jobs: list[tuple[float, PdfPage, str]] = []
    for page in pages:
        text_area = float(page.text_area_ratio or 0.0)
        image_area = float(page.image_area_ratio or 0.0)
        sparse_slide = page.cjk_char_count < 450 and (bool(page.headings) or image_area > 0.12)
        if page.page in useful_pages:
            continue
        if image_area < 0.18 and not (sparse_slide and text_area < 0.55):
            continue
        role = _page_visual_role(page.text)
        score = min(4.0, image_area * 9.0) + (1.0 if sparse_slide else 0.0)
        if page.headings:
            score += 0.5
        page_render_jobs.append((score, page, role))

    page_render_jobs.sort(key=lambda item: item[0], reverse=True)
    for score, page, role in page_render_jobs[:60]:
        rendered = _render_page_png(doc, page.page, renders_dir)
        if rendered is None:
            continue
        render_path, width, height = rendered
        heading = _first_heading(page)
        candidates.append(VisualCandidate(
            candidate_id=f"vis_p{page.page:04d}_page_render",
            page=page.page,
            page_render_path=render_path,
            bbox=(0.0, 0.0, float(page.width or 0.0), float(page.height or 0.0)),
            nearby_heading=heading,
            nearby_text=_short_nearby_text(page.text),
            image_width=width,
            image_height=height,
            image_size=(width, height),
            page_area_ratio=1.0,
            visual_role=role,
            salience_score=round(max(0.0, score), 3),
            is_decorative=False,
            likely_useful=True,
            source_quote=_source_quote(page.text, heading, None),
        ))

    candidates.sort(key=lambda c: (c.page, -c.salience_score, c.candidate_id))
    return candidates


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
    ensure_dir(paths.page_renders_dir)

    ingest_cfg = config.get("ingest", {})
    llm_cfg = config.get("llm", {})

    ocr_enabled = bool(ingest_cfg.get("ocr_fallback", False))
    ocr_min_chars = int(ingest_cfg.get("ocr_min_chars", 50))
    ocr_lang = str(ingest_cfg.get("ocr_lang", "chi_tra+chi_sim+eng"))
    ocr_dpi = int(ingest_cfg.get("ocr_dpi", 200))

    pdf_sha = sha256_file(pdf_path)
    # Capture the OCR environment, not just availability — installing a
    # missing language pack (e.g. `apt install tesseract-ocr-chi-tra`)
    # has to invalidate the cache, otherwise a first run that produced
    # empty pages would be served back on the next call instead of
    # re-OCRing with the new pack.
    ocr_env_sig: dict[str, Any] | None = None
    if ocr_enabled:
        ocr_available = ocr_mod.is_available()
        ocr_env_sig = {
            "available": ocr_available,
            "missing_langs": (
                list(ocr_mod.missing_languages(ocr_lang))
                if ocr_available
                else []
            ),
        }
    # Bump to v4: visual inventory + optional source page renders are
    # now part of chunked_sources.json. (v3 folded OCR environment into
    # the hash; v2 added text_source / bbox / caption_hint.)
    input_hash = hash_inputs(
        "ingest_v4_visual_inventory", pdf_sha, ingest_cfg,
        llm_cfg.get("max_chunk_chars"),
        ocr_env_sig,
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
                page_images: list[PdfImage] = []
                if extract_images:
                    page_images = _extract_page_images(
                        page, doc, pno,
                        images_dir=paths.images_dir,
                        min_pixels=min_pixels, img_fmt=img_fmt,
                    )
                    images.extend(page_images)
                page_w = float(page.rect.width)
                page_h = float(page.rect.height)
                pages.append(PdfPage(
                    page=pno + 1,
                    text=text,
                    cjk_char_count=cjk_char_count(text),
                    headings=headings,
                    text_source=source,
                    width=page_w,
                    height=page_h,
                    text_area_ratio=_page_text_area_ratio(page),
                    image_area_ratio=_page_image_area_ratio(page_images, page_w, page_h),
                ))

            visual_inventory = _build_visual_inventory(
                pages=pages, images=images, doc=doc,
                renders_dir=paths.page_renders_dir,
            )
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
                visual_inventory=visual_inventory,
            )
            atomic_write_json(paths.chunked_json, sources.model_dump(mode="json"))
            ocr_status = {
                "ocr_enabled": ocr_enabled,
                "ocr_available": ocr_mod.is_available() if ocr_enabled else False,
            }
            if ocr_enabled and ocr_mod.is_available():
                missing = ocr_mod.missing_languages(ocr_lang)
                if missing:
                    # Surface in DB + a log line so the user can fix it
                    # before sinking time into a long run that produces
                    # empty pages.
                    ocr_status["missing_ocr_langs"] = list(missing)
                    db.log_error(
                        "ingest",
                        f"OCR enabled but Tesseract is missing language pack(s): "
                        f"{missing!r}. Install via your OS package manager "
                        f"(e.g. `apt install tesseract-ocr-chi-tra`) — OCR will "
                        f"return empty strings until then.",
                    )
            db.register_artifact(paths.chunked_json, stage="ingest",
                                 media_type="application/json",
                                 provenance={"source_pdf": str(pdf_path),
                                             "pdf_sha256": pdf_sha,
                                             **ocr_status})
            for im in images:
                db.register_artifact(im.path, stage="ingest",
                                     media_type=f"image/{img_fmt}",
                                     provenance={"source_pdf": str(pdf_path),
                                                 "page": im.page,
                                                 "pdf_sha256": pdf_sha,
                                                 "bbox": im.bbox,
                                                 "caption_hint": im.caption_hint},
                                     compute_sha=False)
            registered_renders: set[str] = set()
            for vc in sources.visual_inventory:
                render_path = vc.crop_path or vc.page_render_path
                if not render_path or render_path in registered_renders:
                    continue
                registered_renders.add(render_path)
                db.register_artifact(
                    render_path, stage="ingest", media_type="image/png",
                    provenance={"source_pdf": str(pdf_path),
                                "page": vc.page,
                                "pdf_sha256": pdf_sha,
                                "visual_role": vc.visual_role,
                                "candidate_id": vc.candidate_id},
                    compute_sha=False,
                )
        finally:
            doc.close()
        db.finish_stage("ingest", outputs)
        return sources
    except Exception as e:
        db.fail_stage("ingest", repr(e))
        db.log_error("ingest", str(e))
        raise
