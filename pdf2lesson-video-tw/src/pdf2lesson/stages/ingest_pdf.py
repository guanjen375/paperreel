"""Stage 1 — ingest PDF -> chunked_sources.json (+ images dir).

Uses PyMuPDF (fitz). For each page we capture text, headings, CJK char count,
and (optionally) embedded raster images.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

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

    pdf_sha = sha256_file(pdf_path)
    input_hash = hash_inputs("ingest_v1", pdf_sha, ingest_cfg, llm_cfg.get("max_chunk_chars"))
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
                raw_text = page.get_text("text") or ""
                text = normalise_text(raw_text)
                headings = extract_headings(text, max_lines=80)
                heading_count += len(headings)
                pages.append(PdfPage(
                    page=pno + 1,
                    text=text,
                    cjk_char_count=cjk_char_count(text),
                    headings=headings,
                ))
                if not extract_images:
                    continue
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
                        img_path = paths.images_dir / f"p{pno + 1:04d}_x{xref}_{sha[:8]}.{ext}"
                        atomic_write_bytes(img_path, data)
                        images.append(PdfImage(
                            image_id=f"img_p{pno + 1}_{xref}",
                            page=pno + 1,
                            path=str(img_path),
                            width=pix.width,
                            height=pix.height,
                            pixel_count=pixels,
                            sha256=sha,
                        ))
                    finally:
                        pix = None  # release native handle promptly

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
                                             "pdf_sha256": pdf_sha})
            for im in images:
                db.register_artifact(im.path, stage="ingest",
                                     media_type=f"image/{img_fmt}",
                                     provenance={"source_pdf": str(pdf_path),
                                                 "page": im.page,
                                                 "pdf_sha256": pdf_sha},
                                     compute_sha=False)
        finally:
            doc.close()
        db.finish_stage("ingest", outputs)
        return sources
    except Exception as e:
        db.fail_stage("ingest", repr(e))
        db.log_error("ingest", str(e))
        raise
