"""Quality report should flag PDF pages that came back empty or OCR'd.

Without this, a scanned-style PDF without the [ocr] extra installed
silently produces a near-empty video — the pipeline doesn't fail, the
quality report doesn't complain, and the user only notices after
watching `final.mp4`. These tests pin the surfaces.
"""
from __future__ import annotations

import json
from pathlib import Path

from paperreel.io_utils import atomic_write_json
from paperreel.models import (ChunkedSources, PdfChunk, PdfPage, Scene,
                                SceneGraph, SceneStatus, VisualType)
from paperreel.stages import quality_check
from paperreel.state import StateDB


def _scene(scene_id: str, *, page: int = 1) -> Scene:
    return Scene(
        scene_id=scene_id, chapter_id="ch_001", title="t",
        source_pages=[page],
        narration_text_zh_tw="這是一段測試旁白，長度足夠通過 verbatim 檢查。" * 3,
        visual_type=VisualType.bullet_card,
        estimated_duration_sec=25.0,
        actual_duration_sec=25.0,
        audio_path="/tmp/fake.wav",
        subtitle_path="/tmp/fake.srt",
        visual_asset_paths=["/tmp/fake.png"],
        status=SceneStatus.rendered,
        input_hash="hash",
    )


def _write_chunked_and_graph(
    project_dir: Path,
    *,
    pages: list[PdfPage],
    scenes: list[Scene],
    target_minutes: float = 5.0,
) -> None:
    """Hand-build chunked_sources.json + scene_graph.json so the quality
    stage runs without needing the real ingest pipeline."""
    inter = project_dir / "intermediate"
    inter.mkdir(parents=True, exist_ok=True)
    cs = ChunkedSources(
        source_pdf="dummy.pdf", pdf_sha256="x" * 64,
        page_count=len(pages),
        cjk_char_count=sum(p.cjk_char_count for p in pages),
        image_count=0, heading_count=0,
        estimated_density=0.0,
        pages=pages,
        chunks=[PdfChunk(chunk_id="chunk_0001", start_page=1,
                          end_page=len(pages),
                          text=" ".join(p.text for p in pages),
                          cjk_char_count=sum(p.cjk_char_count for p in pages))],
        images=[],
    )
    atomic_write_json(inter / "chunked_sources.json", cs.model_dump(mode="json"))
    g = SceneGraph(project="t", target_minutes=target_minutes, scenes=scenes)
    atomic_write_json(inter / "scene_graph.json", g.model_dump(mode="json"))


def test_warns_when_any_pages_are_empty(
    project_dir: Path, test_cfg: dict
) -> None:
    pages = [
        PdfPage(page=1, text="第一頁有內容。", cjk_char_count=8,
                 text_source="text"),
        PdfPage(page=2, text="", cjk_char_count=0,
                 text_source="empty"),
    ]
    _write_chunked_and_graph(project_dir, pages=pages,
                              scenes=[_scene("ch_001_sc_001", page=1)])
    db = StateDB(project_dir / "state.sqlite")
    report = quality_check.run(project_root=project_dir, db=db,
                                config=test_cfg)
    db.close()
    codes = [i.code for i in report.issues]
    assert "empty_pdf_pages" in codes
    msg = next(i for i in report.issues if i.code == "empty_pdf_pages").message
    assert "2" in msg, f"expected the empty page number in message, got {msg!r}"


def test_info_message_when_pages_used_ocr(
    project_dir: Path, test_cfg: dict
) -> None:
    pages = [
        PdfPage(page=1, text="OCR 抽到的內容。", cjk_char_count=8,
                 text_source="ocr"),
    ]
    _write_chunked_and_graph(project_dir, pages=pages,
                              scenes=[_scene("ch_001_sc_001", page=1)])
    db = StateDB(project_dir / "state.sqlite")
    report = quality_check.run(project_root=project_dir, db=db,
                                config=test_cfg)
    db.close()
    ocr_issue = next((i for i in report.issues if i.code == "ocr_pages"), None)
    assert ocr_issue is not None
    assert ocr_issue.severity == "info"


def test_warns_when_empty_page_ratio_high(
    project_dir: Path, test_cfg: dict
) -> None:
    """A PDF where most pages came back empty almost always means a
    scanned-style document without OCR installed."""
    pages = (
        [PdfPage(page=i, text="", cjk_char_count=0, text_source="empty")
         for i in range(1, 5)] +
        [PdfPage(page=5, text="只有第五頁有字。", cjk_char_count=8,
                  text_source="text")]
    )
    _write_chunked_and_graph(project_dir, pages=pages,
                              scenes=[_scene("ch_001_sc_001", page=5)])
    db = StateDB(project_dir / "state.sqlite")
    cfg = {**test_cfg,
           "quality": {**test_cfg["quality"], "empty_pages_warn_pct": 20.0}}
    report = quality_check.run(project_root=project_dir, db=db, config=cfg)
    db.close()
    codes = [i.code for i in report.issues]
    assert "many_empty_pages" in codes


def test_no_pdf_page_issues_for_clean_digital_pdf(
    project_dir: Path, test_cfg: dict
) -> None:
    pages = [
        PdfPage(page=i, text=f"第 {i} 頁有正常的內容。", cjk_char_count=10,
                 text_source="text")
        for i in range(1, 5)
    ]
    _write_chunked_and_graph(project_dir, pages=pages,
                              scenes=[_scene("ch_001_sc_001", page=1)])
    db = StateDB(project_dir / "state.sqlite")
    report = quality_check.run(project_root=project_dir, db=db, config=test_cfg)
    db.close()
    codes = {i.code for i in report.issues}
    assert "empty_pdf_pages" not in codes
    assert "ocr_pages" not in codes
    assert "many_empty_pages" not in codes
