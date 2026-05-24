"""Stage 10 — quality_report.json.

Checks:
  - final video within ±tolerance% of target
  - every scene has source_pages
  - every scene has audio + (subtitle if requested)
  - missing assets are listed
  - flag verbatim-overlap > warn_verbatim_pct
  - PDF pages that came back empty (no extractable text and no OCR
    result) — these vanish silently into the script otherwise
  - PDF pages that used OCR — informational, surfaces accuracy risk
  - high empty-page ratio — likely scanned PDF needing the [ocr]
    extra installed
"""
from __future__ import annotations

from pathlib import Path

from ..hashing import hash_inputs
from ..io_utils import atomic_write_json, read_json
from ..models import (ChunkedSources, QualityIssue, QualityReport,
                      SceneGraph, SceneStatus)
from ..renderers.ffmpeg_renderer import have_ffmpeg, probe_duration_seconds
from ..state import StateDB
from ..utils.text_cleaning import verbatim_overlap_ratio


def paths_for(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    return {
        "scene_graph": root / "intermediate" / "scene_graph.json",
        "chunked": root / "intermediate" / "chunked_sources.json",
        "final": root / "outputs" / "final.mp4",
        "report": root / "outputs" / "quality_report.json",
    }


def run(*, project_root: str | Path, db: StateDB, config: dict) -> QualityReport:
    p = paths_for(project_root)
    graph = SceneGraph.model_validate(read_json(p["scene_graph"]))
    sources = ChunkedSources.model_validate(read_json(p["chunked"]))

    qcfg = config.get("quality", {})
    tol_pct = float(qcfg.get("duration_tolerance_pct", 8.0))
    warn_v_pct = float(qcfg.get("warn_verbatim_pct", 5.0))
    empty_warn_pct = float(qcfg.get("empty_pages_warn_pct", 20.0))

    input_hash = hash_inputs("quality_v1", graph.project,
                             [s.scene_id for s in graph.scenes],
                             [s.status.value for s in graph.scenes],
                             p["final"].exists(), len(sources.pages))
    db.start_stage("quality", input_hash)

    try:
        issues: list[QualityIssue] = []
        failed_ids: list[str] = []
        rendered = 0

        # Surface PDF-side coverage issues first: empty pages disappear
        # from the script entirely, and a scanned-style PDF with no OCR
        # installed will quietly produce a near-empty video. Both are
        # easy to miss until the user previews `final.mp4`.
        empty_pages = [p.page for p in sources.pages
                        if p.text_source == "empty"]
        ocr_pages = [p.page for p in sources.pages
                      if p.text_source == "ocr"]
        if empty_pages:
            preview = empty_pages[:20]
            more = "" if len(empty_pages) <= 20 else f", ... ({len(empty_pages)} total)"
            issues.append(QualityIssue(
                severity="warning", code="empty_pdf_pages",
                message=(
                    f"{len(empty_pages)} PDF pages had no extractable text "
                    f"and no OCR result: {preview}{more}. Install the "
                    "`[ocr]` extra and a tesseract language pack if this "
                    "is a scanned PDF."
                ),
            ))
        if ocr_pages:
            preview = ocr_pages[:20]
            more = "" if len(ocr_pages) <= 20 else f", ... ({len(ocr_pages)} total)"
            issues.append(QualityIssue(
                severity="info", code="ocr_pages",
                message=(
                    f"{len(ocr_pages)} PDF pages used OCR fallback: "
                    f"{preview}{more}. Review the narration for those pages "
                    "— OCR can mis-read footnotes and equations."
                ),
            ))
        if sources.pages:
            empty_ratio_pct = 100.0 * len(empty_pages) / len(sources.pages)
            if empty_ratio_pct >= empty_warn_pct:
                issues.append(QualityIssue(
                    severity="warning", code="many_empty_pages",
                    message=(
                        f"{empty_ratio_pct:.0f}% of PDF pages produced no "
                        f"text (limit {empty_warn_pct:.0f}%). The video "
                        "likely under-covers the source."
                    ),
                ))

        pdf_full_text = " ".join(pg.text for pg in sources.pages)

        for sc in graph.scenes:
            if sc.status == SceneStatus.rendered:
                rendered += 1
            if sc.status == SceneStatus.failed:
                failed_ids.append(sc.scene_id)
                issues.append(QualityIssue(
                    severity="error", code="scene_failed",
                    message=f"scene failed: {sc.last_error or 'unknown'}",
                    scene_id=sc.scene_id,
                ))
                continue
            if not sc.source_pages:
                issues.append(QualityIssue(
                    severity="error", code="missing_source_pages",
                    message="scene has no source_pages — provenance broken",
                    scene_id=sc.scene_id,
                ))
            if not sc.audio_path:
                issues.append(QualityIssue(
                    severity="error", code="missing_audio",
                    message="audio asset missing", scene_id=sc.scene_id,
                ))
            if not sc.subtitle_path:
                issues.append(QualityIssue(
                    severity="warning", code="missing_subtitle",
                    message="subtitle file missing", scene_id=sc.scene_id,
                ))
            if not sc.visual_asset_paths:
                issues.append(QualityIssue(
                    severity="error", code="missing_visual",
                    message="visual asset missing", scene_id=sc.scene_id,
                ))
            ratio = verbatim_overlap_ratio(sc.narration_text_zh_tw, pdf_full_text)
            if ratio * 100.0 > warn_v_pct:
                issues.append(QualityIssue(
                    severity="warning", code="verbatim_overlap",
                    message=f"講稿 {ratio*100:.1f}% 與原文逐字重疊 (上限 {warn_v_pct:.1f}%)",
                    scene_id=sc.scene_id,
                ))

        final_duration = 0.0
        final_path: str | None = None
        if p["final"].exists():
            final_path = str(p["final"])
            if have_ffmpeg(config.get("renderer", {}).get("ffprobe_binary", "ffprobe")):
                try:
                    final_duration = probe_duration_seconds(p["final"])
                except Exception:
                    final_duration = sum((s.actual_duration_sec or s.estimated_duration_sec)
                                         for s in graph.scenes if s.status == SceneStatus.rendered)
            else:
                final_duration = sum((s.actual_duration_sec or s.estimated_duration_sec)
                                     for s in graph.scenes if s.status == SceneStatus.rendered)

        target_sec = graph.target_minutes * 60.0
        if final_duration > 0 and target_sec > 0:
            delta_pct = abs(final_duration - target_sec) / target_sec * 100.0
            if delta_pct > tol_pct:
                issues.append(QualityIssue(
                    severity="warning", code="duration_off_target",
                    message=f"final {final_duration/60:.1f} min vs target {graph.target_minutes:.1f} min ({delta_pct:.1f}% off)",
                ))

        report = QualityReport(
            project=graph.project,
            target_minutes=graph.target_minutes,
            final_duration_sec=final_duration,
            final_video_path=final_path,
            scene_count=len(graph.scenes),
            rendered_scene_count=rendered,
            failed_scene_ids=failed_ids,
            issues=issues,
        )
        atomic_write_json(p["report"], report.model_dump(mode="json"))
        db.register_artifact(p["report"], stage="quality", media_type="application/json")
        db.finish_stage("quality", [str(p["report"])])
        return report
    except Exception as e:
        db.fail_stage("quality", repr(e))
        db.log_error("quality", str(e))
        raise
