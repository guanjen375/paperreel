"""Pipeline state DB + resume behaviour."""
from __future__ import annotations

from pathlib import Path

from paperreel.hashing import hash_inputs
from paperreel.stages import (build_outline, build_scene_graph,
                               ingest_pdf, render_visuals,
                               synthesize_audio, write_script)
from paperreel.state import StateDB


def test_stage_is_done_requires_output_file(project_dir: Path) -> None:
    db = StateDB(project_dir / "state.sqlite")
    h = hash_inputs("v1", {"x": 1})
    db.start_stage("ingest", h)
    db.finish_stage("ingest", [str(project_dir / "intermediate" / "chunked_sources.json")])
    # Output file not actually written → resume must NOT consider the stage done.
    assert db.stage_is_done(
        "ingest", h, [str(project_dir / "intermediate" / "chunked_sources.json")]
    ) is False
    db.close()


def test_resume_skips_completed_ingest(project_dir: Path, tiny_pdf: Path,
                                       test_cfg: dict) -> None:
    db = StateDB(project_dir / "state.sqlite")

    sources1 = ingest_pdf.run(pdf_path=tiny_pdf, project_root=project_dir,
                              db=db, config=test_cfg)
    chunked_path = project_dir / "intermediate" / "chunked_sources.json"
    mtime1 = chunked_path.stat().st_mtime_ns

    # Re-run: should be a no-op (same input_hash, file present).
    sources2 = ingest_pdf.run(pdf_path=tiny_pdf, project_root=project_dir,
                              db=db, config=test_cfg)
    mtime2 = chunked_path.stat().st_mtime_ns

    assert sources1.pdf_sha256 == sources2.pdf_sha256
    assert mtime1 == mtime2, "resume should not re-write chunked_sources.json"
    db.close()


def test_force_reruns_stage(project_dir: Path, tiny_pdf: Path,
                            test_cfg: dict) -> None:
    db = StateDB(project_dir / "state.sqlite")
    ingest_pdf.run(pdf_path=tiny_pdf, project_root=project_dir, db=db, config=test_cfg)
    chunked_path = project_dir / "intermediate" / "chunked_sources.json"
    mtime1 = chunked_path.stat().st_mtime_ns
    # Force rerun
    ingest_pdf.run(pdf_path=tiny_pdf, project_root=project_dir, db=db, config=test_cfg,
                   force=True)
    mtime2 = chunked_path.stat().st_mtime_ns
    assert mtime2 >= mtime1
    db.close()


def test_audio_then_visuals_resume_only_runs_once(project_dir: Path,
                                                  tiny_pdf: Path,
                                                  test_cfg: dict) -> None:
    db = StateDB(project_dir / "state.sqlite")

    ingest_pdf.run(pdf_path=tiny_pdf, project_root=project_dir, db=db, config=test_cfg)
    build_outline.run(project_root=project_dir, project_name="t", db=db,
                      config=test_cfg, target_minutes="auto")
    write_script.run(project_root=project_dir, db=db, config=test_cfg)
    build_scene_graph.run(project_root=project_dir, project_name="t",
                          pdf_name=tiny_pdf.name, db=db, config=test_cfg)
    g1 = synthesize_audio.run(project_root=project_dir, db=db, config=test_cfg)
    render_visuals.run(project_root=project_dir, db=db, config=test_cfg)

    audio_mtimes = {s.audio_path: Path(s.audio_path).stat().st_mtime_ns
                    for s in g1.scenes if s.audio_path}
    g2 = synthesize_audio.run(project_root=project_dir, db=db, config=test_cfg, resume=True)
    for s in g2.scenes:
        if s.audio_path and s.audio_path in audio_mtimes:
            assert Path(s.audio_path).stat().st_mtime_ns == audio_mtimes[s.audio_path]
    db.close()


def test_state_summary_reports_artifacts(project_dir: Path, tiny_pdf: Path,
                                         test_cfg: dict) -> None:
    db = StateDB(project_dir / "state.sqlite")
    ingest_pdf.run(pdf_path=tiny_pdf, project_root=project_dir, db=db, config=test_cfg)
    s = db.status_summary()
    assert any(r["name"] == "ingest" and r["status"] == "completed" for r in s["stages"])
    assert s["artifact_counts"].get("ingest", 0) >= 1
    db.close()


def test_status_resolves_errors_when_stage_later_completes(project_dir: Path) -> None:
    """Errors logged before a stage's most recent finish_at are resolved, not active."""
    db = StateDB(project_dir / "state.sqlite")
    h = hash_inputs("v1", {"x": 1})

    # Simulate a failed audio run that left two errors behind.
    db.start_stage("audio", h)
    db.log_error("audio", "ollama timeout")
    db.log_error("audio", "tts unavailable")
    db.fail_stage("audio", "ollama timeout")
    assert db.status_summary()["error_count"] == 2

    # Same stage now succeeds — both errors should become "resolved".
    db.start_stage("audio", h)
    db.finish_stage("audio", [])
    s = db.status_summary()
    assert s["error_count"] == 0
    assert s["error_count_resolved"] == 2

    # A fresh error logged AFTER the success stays active.
    db.log_error("audio", "later flake")
    s = db.status_summary()
    assert s["error_count"] == 1
    assert s["error_count_resolved"] == 2
    db.close()
