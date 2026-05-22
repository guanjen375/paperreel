"""Typer CLI — one command per pipeline stage plus the orchestrator `all`."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    # Rich prints non-ASCII glyphs (✓ ✗ 繁中) that crash on legacy cp* code-page
    # consoles (e.g. PowerShell 5.1 with cp950). Force the IO streams to UTF-8
    # so writes don't raise UnicodeEncodeError mid-pipeline.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .io_utils import atomic_write_json, ensure_dir
from .models import ProjectMeta
from .stages import (build_outline, build_scene_graph, build_subtitles,
                     concat_final, ingest_pdf, quality_check, render_segments,
                     render_visuals, synthesize_audio, write_script)
from .state import StateDB

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="PDF -> 繁體中文教學影片 pipeline.")
console = Console()


# ---------- helpers ----------

def _project_paths(project: str | Path) -> dict[str, Path]:
    root = Path(project).resolve()
    return {
        "root": root,
        "intermediate": root / "intermediate",
        "assets": root / "assets",
        "outputs": root / "outputs",
        "logs": root / "logs",
        "meta": root / "project.json",
        "db": root / "state.sqlite",
        "config": root / "config.yaml",
    }


def _ensure_project(project: str | Path, *,
                    source_pdf: str | None = None,
                    overlay: str | None = None) -> tuple[dict[str, Path], dict, StateDB, ProjectMeta]:
    p = _project_paths(project)
    for k in ("root", "intermediate", "assets", "outputs", "logs"):
        ensure_dir(p[k])
    if not p["meta"].exists():
        meta = ProjectMeta(name=p["root"].name, root=str(p["root"]),
                           source_pdf=source_pdf, config_overlay=overlay)
        atomic_write_json(p["meta"], meta.model_dump(mode="json"))
    else:
        meta = ProjectMeta.model_validate(json.loads(p["meta"].read_text(encoding="utf-8")))
        if source_pdf and not meta.source_pdf:
            meta = meta.model_copy(update={"source_pdf": source_pdf})
            atomic_write_json(p["meta"], meta.model_dump(mode="json"))
    cfg = load_config(overlay or meta.config_overlay)
    db = StateDB(p["db"])
    db.upsert_project(meta.name, str(meta.root),
                      source_pdf=meta.source_pdf,
                      config_overlay=meta.config_overlay,
                      meta=meta.model_dump(mode="json"))
    return p, cfg, db, meta


def _check_budget(start_ts: float, max_hours: float) -> None:
    if max_hours <= 0:
        return
    elapsed_hours = (time.time() - start_ts) / 3600.0
    if elapsed_hours >= max_hours:
        raise typer.Exit(code=2)


def _abort(msg: str) -> None:
    console.print(f"[red]✗ {msg}[/red]")
    raise typer.Exit(code=1)


# ---------- commands ----------

@app.command()
def init(
    project_dir: str = typer.Argument(..., help="Path to create / re-use as project root"),
    overlay: Optional[str] = typer.Option(None, "--config", "-c",
                                          help="Optional config overlay (name in configs/ or path)"),
) -> None:
    """Create project_dir with intermediate/, assets/, outputs/, project.json, state.sqlite."""
    p, cfg, db, meta = _ensure_project(project_dir, overlay=overlay)
    console.print(f"[green]✓[/green] initialised project: {p['root']}")
    console.print(f"  config overlay: {meta.config_overlay or '(default)'}")


@app.command()
def ingest(
    pdf: str = typer.Argument(..., help="Source PDF path"),
    project: str = typer.Option(..., "--project", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Extract text/images/chunks -> intermediate/chunked_sources.json."""
    if not Path(pdf).exists():
        _abort(f"PDF not found: {pdf}")
    p, cfg, db, meta = _ensure_project(project, source_pdf=str(Path(pdf).resolve()),
                                       overlay=config)
    sources = ingest_pdf.run(pdf_path=pdf, project_root=p["root"], db=db,
                             config=cfg, force=force)
    console.print(f"[green]✓[/green] ingested: {sources.page_count} pages, "
                  f"{sources.cjk_char_count} CJK chars, {sources.image_count} images")


@app.command()
def plan(
    project: str = typer.Option(..., "--project", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    target_minutes: str = typer.Option("auto", "--target-minutes"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Generate lesson_outline.json + duration_plan.json."""
    p, cfg, db, _ = _ensure_project(project, overlay=config)
    outline = build_outline.run(project_root=p["root"], project_name=Path(project).name,
                                db=db, config=cfg,
                                target_minutes=target_minutes, force=force)
    console.print(f"[green]✓[/green] outline: {len(outline.chapters)} chapters, "
                  f"target {outline.target_minutes:.1f} min")


@app.command()
def script(
    project: str = typer.Option(..., "--project", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Generate script.json (list of ScriptScene)."""
    p, cfg, db, _ = _ensure_project(project, overlay=config)
    sc = write_script.run(project_root=p["root"], db=db, config=cfg, force=force)
    console.print(f"[green]✓[/green] script: {len(sc.scenes)} scenes, "
                  f"~{sc.total_estimated_minutes:.1f} min")


@app.command()
def scenes(
    project: str = typer.Option(..., "--project", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Build scene_graph.json from script (inserts recap scenes)."""
    p, cfg, db, meta = _ensure_project(project, overlay=config)
    pdf_name = Path(meta.source_pdf).name if meta.source_pdf else "source.pdf"
    g = build_scene_graph.run(project_root=p["root"], project_name=meta.name,
                              pdf_name=pdf_name, db=db, config=cfg, force=force)
    console.print(f"[green]✓[/green] scene graph: {len(g.scenes)} scenes")


@app.command()
def audio(
    project: str = typer.Option(..., "--project", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """Synthesize TTS for each scene."""
    p, cfg, db, _ = _ensure_project(project, overlay=config)
    g = synthesize_audio.run(project_root=p["root"], db=db, config=cfg, resume=resume,
                             max_retries=int(cfg.get("runtime", {}).get("scene_retry_max", 2)))
    done = sum(1 for s in g.scenes if s.audio_path)
    console.print(f"[green]✓[/green] audio: {done}/{len(g.scenes)} scenes synthesized")


@app.command()
def visuals(
    project: str = typer.Option(..., "--project", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """Render per-scene PNG visuals."""
    p, cfg, db, _ = _ensure_project(project, overlay=config)
    g = render_visuals.run(project_root=p["root"], db=db, config=cfg, resume=resume)
    done = sum(1 for s in g.scenes if s.visual_asset_paths)
    console.print(f"[green]✓[/green] visuals: {done}/{len(g.scenes)} scenes rendered")


@app.command()
def subtitles(
    project: str = typer.Option(..., "--project", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Build per-scene + full subtitles."""
    p, cfg, db, _ = _ensure_project(project, overlay=config)
    g = build_subtitles.run(project_root=p["root"], db=db, config=cfg)
    console.print(f"[green]✓[/green] subtitles built for {len(g.scenes)} scenes")


@app.command()
def render(
    project: str = typer.Option(..., "--project", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """Mux segments + concat final.mp4 + quality report."""
    p, cfg, db, _ = _ensure_project(project, overlay=config)
    if shutil.which(cfg.get("renderer", {}).get("ffmpeg_binary", "ffmpeg")) is None:
        _abort("ffmpeg not on PATH. Install ffmpeg or run with --no-render.")
    plan, g = render_segments.run(project_root=p["root"], db=db, config=cfg, resume=resume)
    final = concat_final.run(project_root=p["root"], db=db, config=cfg)
    report = quality_check.run(project_root=p["root"], db=db, config=cfg)
    console.print(f"[green]✓[/green] final: {final}")
    console.print(f"  rendered scenes: {report.rendered_scene_count}/{report.scene_count}, "
                  f"final duration: {report.final_duration_sec/60:.1f} min, "
                  f"issues: {len(report.issues)}")


@app.command(name="all")
def run_all(
    pdf: str = typer.Argument(..., help="Source PDF path"),
    project: str = typer.Option(..., "--project", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    target_minutes: str = typer.Option("auto", "--target-minutes"),
    max_hours: float = typer.Option(10.0, "--max-hours"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    force_stage: Optional[str] = typer.Option(
        None, "--force-stage",
        help="Comma-separated stage names to force re-run (e.g. 'plan,script')"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Force mock providers + tiny resolution"),
    skip_render: bool = typer.Option(False, "--skip-render",
                                     help="Stop after subtitles (skip segments/concat/quality)"),
) -> None:
    """Run the full pipeline. Each stage is resumable via SQLite state."""
    if not Path(pdf).exists():
        _abort(f"PDF not found: {pdf}")
    overlay = config
    if dry_run:
        # `--dry-run` forces the dryrun overlay regardless of user choice.
        overlay = "dryrun.yaml"
    p, cfg, db, meta = _ensure_project(project, source_pdf=str(Path(pdf).resolve()),
                                       overlay=overlay)
    forced = {s.strip() for s in (force_stage or "").split(",") if s.strip()}
    start = time.time()
    pdf_name = Path(meta.source_pdf or pdf).name

    ingest_pdf.run(pdf_path=pdf, project_root=p["root"], db=db, config=cfg,
                   force="ingest" in forced)
    _check_budget(start, max_hours)

    build_outline.run(project_root=p["root"], project_name=meta.name, db=db, config=cfg,
                      target_minutes=target_minutes, force="plan" in forced)
    _check_budget(start, max_hours)

    write_script.run(project_root=p["root"], db=db, config=cfg,
                     force="script" in forced)
    _check_budget(start, max_hours)

    build_scene_graph.run(project_root=p["root"], project_name=meta.name,
                          pdf_name=pdf_name, db=db, config=cfg,
                          force="scenes" in forced)
    _check_budget(start, max_hours)

    synthesize_audio.run(project_root=p["root"], db=db, config=cfg, resume=resume,
                         max_retries=int(cfg.get("runtime", {}).get("scene_retry_max", 2)))
    _check_budget(start, max_hours)

    render_visuals.run(project_root=p["root"], db=db, config=cfg, resume=resume)
    _check_budget(start, max_hours)

    build_subtitles.run(project_root=p["root"], db=db, config=cfg)
    _check_budget(start, max_hours)

    if skip_render:
        console.print("[yellow]skip-render: stopping after subtitles[/yellow]")
        return

    if shutil.which(cfg.get("renderer", {}).get("ffmpeg_binary", "ffmpeg")) is None:
        console.print("[yellow]! ffmpeg not on PATH — segments/concat/quality skipped.[/yellow]")
        console.print("  install ffmpeg then re-run: paperreel render --project " + str(p["root"]))
        return

    render_segments.run(project_root=p["root"], db=db, config=cfg, resume=resume)
    _check_budget(start, max_hours)
    final = concat_final.run(project_root=p["root"], db=db, config=cfg)
    report = quality_check.run(project_root=p["root"], db=db, config=cfg)
    console.print(f"[green]✓[/green] final video: {final}")
    console.print(f"  rendered {report.rendered_scene_count}/{report.scene_count} scenes "
                  f"({report.final_duration_sec/60:.1f} min, {len(report.issues)} issues)")


@app.command()
def status(
    project: str = typer.Option(..., "--project", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Print the current pipeline state from SQLite."""
    p, _cfg, db, meta = _ensure_project(project, overlay=config)
    s = db.status_summary()
    console.print(f"[bold]project[/bold]: {meta.name}  ({p['root']})")

    stage_tbl = Table(title="Stages", show_lines=False)
    stage_tbl.add_column("name"); stage_tbl.add_column("status")
    stage_tbl.add_column("finished_at"); stage_tbl.add_column("input_hash", overflow="fold")
    order = ["ingest", "plan", "script", "scenes", "audio", "visuals",
             "subtitles", "segments", "concat", "quality"]
    by_name = {r["name"]: r for r in s["stages"]}
    for n in order:
        r = by_name.get(n)
        stage_tbl.add_row(n,
                          (r and r["status"]) or "-",
                          (r and (r["finished_at"] or "")) or "-",
                          (r and (r["input_hash"] or "")[:12]) or "-")
    console.print(stage_tbl)

    scene_tbl = Table(title="Scenes by status")
    scene_tbl.add_column("status"); scene_tbl.add_column("count", justify="right")
    for k, v in sorted(s["scene_status_counts"].items()):
        scene_tbl.add_row(k, str(v))
    console.print(scene_tbl)

    console.print(f"artifacts: { s['artifact_counts'] }")
    console.print(f"errors: { s['error_count'] }")


@app.command(name="retry-failed")
def retry_failed(
    project: str = typer.Option(..., "--project", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Reset failed scenes -> pending so the next audio/visual/render pass retries them."""
    p, cfg, db, _ = _ensure_project(project, overlay=config)
    n = db.reset_failed_scenes()
    console.print(f"[green]✓[/green] reset {n} failed scenes to pending")


if __name__ == "__main__":
    app()
