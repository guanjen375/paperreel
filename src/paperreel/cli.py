"""Typer CLI — ``paperreel <pdf>`` runs the full pipeline; subcommands target individual stages."""
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
                     concat_final, ingest_pdf, match_pdf_visuals,
                     quality_check, render_segments, render_visuals,
                     synthesize_audio, write_script)
from .state import StateDB

app = typer.Typer(
    add_completion=False, no_args_is_help=True,
    help=(
        "PDF -> 繁體中文教學影片 pipeline.\n\n"
        "Usage: paperreel <pdf> --project <dir>   (shorthand for `paperreel run …`)\n"
        "Run `paperreel run --help` for full pipeline flags."
    ),
)
console = Console()


# ---------- helpers ----------

def _release_gpu() -> None:
    """Free GPU references between heavy stages.

    We run TTS (coqui-tts / XTTS) and SDXL in the same Python process; if
    TTS-side allocations are still resident when SDXL loads, the SDXL UNet
    forward can deadlock on layer_norm. Calling this between stages is a
    cheap insurance against that.
    """
    import gc
    gc.collect()
    try:
        import torch  # noqa: WPS433
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

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


def _fmt_elapsed(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    return f"{m}m {int(s - m * 60)}s"


def _ollama_daemon_on_terminal() -> str | None:
    """If an ``ollama serve`` is logging to a tty we share, return that tty name.

    We can't silence an external daemon's stderr from inside Python — but we
    can detect when one will spam this run's progress lines and tell the user
    how to fix it (one-shot tip, no auto-action). Returns ``None`` when we
    can't tell or the daemon is logging elsewhere; that's the safe default.
    Linux-only — falls through silently on macOS/Windows where /proc/<pid>/fd
    isn't available.
    """
    import os
    import subprocess
    if not sys.platform.startswith("linux"):
        return None
    try:
        result = subprocess.run(
            ["pgrep", "-f", "ollama serve"],
            capture_output=True, text=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    pids = [p for p in result.stdout.split() if p.isdigit()]
    if not pids:
        return None
    # Collect tty names attached to *our* fds 0/1/2 (any of them tells us
    # which terminal we're on; running under `tee` etc. may detach some).
    our_ttys: set[str] = set()
    for fd in (0, 1, 2):
        try:
            our_ttys.add(os.ttyname(fd))
        except OSError:
            continue
    if not our_ttys:
        return None
    for pid in pids:
        for fd in ("1", "2"):
            try:
                link = os.readlink(f"/proc/{pid}/fd/{fd}")
            except (OSError, PermissionError):
                continue
            if link in our_ttys:
                return link
    return None


class _Pipeline:
    """Stage-by-stage runner with spinner + completion lines.

    Why this exists: the full pipeline can take 10+ minutes; without a
    visible per-stage indicator users can't tell a slow LLM call from
    a hung process. Each ``.stage()`` shows a spinner while the work
    runs and a checkmark + elapsed time + optional summary when it
    finishes. Cached/resumed stages naturally show ~0s elapsed.
    """
    def __init__(self, total: int):
        self.total = total
        self.idx = 0

    def stage(self, label: str, fn, *, summary=None):
        self.idx += 1
        t0 = time.time()
        with console.status(
            f"[cyan][{self.idx}/{self.total}] {label}…[/]", spinner="dots"
        ):
            result = fn()
        elapsed = time.time() - t0
        line = f"[green]✓[/] [{self.idx}/{self.total}] {label}"
        if summary is not None and result is not None:
            try:
                tail = summary(result)
            except Exception:
                tail = ""
            if tail:
                line += f" — {tail}"
        line += f"  [dim]({_fmt_elapsed(elapsed)})[/]"
        console.print(line)
        return result


# ---------- commands ----------

@app.command(name="run")
def run_pipeline(
    pdf: str = typer.Argument(..., help="Source PDF path."),
    project: str = typer.Option(
        ..., "--project", "-p",
        help="Project root — will be created or resumed in place."),
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Config overlay: bundled name (e.g. 'bigvram') or filesystem path."),
    target_minutes: str = typer.Option(
        "auto", "--target-minutes",
        help="'auto' (estimate from PDF length) or an integer (force, e.g. '15')."),
    max_hours: float = typer.Option(
        10.0, "--max-hours",
        help="Wall-time budget; pipeline halts when exceeded (next run resumes)."),
    force_stage: Optional[str] = typer.Option(
        None, "--force-stage",
        help="Comma-separated stages to force re-run, e.g. 'plan,script'."),
    skip_render: bool = typer.Option(
        False, "--skip-render",
        help="Stop after subtitles; skip mp4 mux/concat/quality."),
) -> None:
    """Generate a 繁中 教學影片 from a PDF.

    Auto-resumes: re-running on the same --project picks up from the last
    completed stage. Use --force-stage to redo specific stages.

    Tip: ``paperreel <pdf>`` is a shorthand — the ``main_entry`` wrapper
    inserts ``run`` for you when the first arg isn't a known subcommand.
    """
    _run_full_pipeline(
        pdf=pdf, project=project, config=config,
        target_minutes=target_minutes, max_hours=max_hours,
        force_stage=force_stage, skip_render=skip_render,
    )


def _run_full_pipeline(*, pdf: str, project: str, config: Optional[str],
                      target_minutes: str, max_hours: float,
                      force_stage: Optional[str], skip_render: bool) -> None:
    if not Path(pdf).exists():
        _abort(f"PDF not found: {pdf}")
    p, cfg, db, meta = _ensure_project(
        project, source_pdf=str(Path(pdf).resolve()), overlay=config,
    )
    forced = {s.strip() for s in (force_stage or "").split(",") if s.strip()}
    start = time.time()
    pdf_name = Path(meta.source_pdf or pdf).name

    ffmpeg_bin = cfg.get("renderer", {}).get("ffmpeg_binary", "ffmpeg")
    will_render = (not skip_render) and (shutil.which(ffmpeg_bin) is not None)
    total = 11 if will_render else 8

    console.print()
    console.print(f"[bold]PDF[/]      {Path(pdf).resolve()}")
    console.print(f"[bold]Project[/]  {p['root']}")
    console.print(f"[bold]Target[/]   {target_minutes} min")
    console.print(f"[bold]Config[/]   {meta.config_overlay or 'default'}")
    console.print(
        f"[bold]Pipeline[/] {total} stages "
        f"({'render to mp4' if will_render else 'no render — stops after subtitles'})"
    )
    if not will_render and not skip_render:
        console.print(
            "[yellow]! ffmpeg not on PATH — final render will be skipped. "
            "Install ffmpeg, then re-run to finish.[/]"
        )
    ollama_tty = _ollama_daemon_on_terminal()
    if ollama_tty:
        # We can silence in-process noise (diffusers etc.) but not an external
        # daemon logging to a shared tty. Give the user a one-shot fix.
        console.print(
            f"[dim]! tip: ollama daemon is logging to this terminal ({ollama_tty}). "
            f"To silence next run, restart it detached, e.g.:[/]\n"
            f"  [dim]pkill -f 'ollama serve' && nohup ollama serve "
            f">/tmp/ollama.log 2>&1 &[/]"
        )
    console.print("─" * 60)

    pl = _Pipeline(total=total)

    pl.stage(
        "解析 PDF",
        lambda: ingest_pdf.run(
            pdf_path=pdf, project_root=p["root"], db=db, config=cfg,
            force="ingest" in forced,
        ),
        summary=lambda s: f"{s.page_count} 頁, {s.cjk_char_count} CJK 字, {s.image_count} 圖",
    )
    _check_budget(start, max_hours)

    pl.stage(
        "規劃章節",
        lambda: build_outline.run(
            project_root=p["root"], project_name=meta.name, db=db, config=cfg,
            target_minutes=target_minutes, force="plan" in forced,
        ),
        summary=lambda o: f"{len(o.chapters)} 章, {o.target_minutes:.1f} 分鐘",
    )
    _check_budget(start, max_hours)

    pl.stage(
        "寫腳本",
        lambda: write_script.run(
            project_root=p["root"], db=db, config=cfg, force="script" in forced,
        ),
        summary=lambda sc: f"{len(sc.scenes)} scenes, ~{sc.total_estimated_minutes:.1f} 分鐘",
    )
    _check_budget(start, max_hours)

    pl.stage(
        "組 scene graph",
        lambda: build_scene_graph.run(
            project_root=p["root"], project_name=meta.name, pdf_name=pdf_name,
            db=db, config=cfg, force="scenes" in forced,
        ),
        summary=lambda g: f"{len(g.scenes)} scenes",
    )
    _check_budget(start, max_hours)

    pl.stage(
        "配對 PDF 圖片",
        lambda: match_pdf_visuals.run(project_root=p["root"], db=db, config=cfg),
        summary=lambda g: (
            f"{sum(1 for s in g.scenes if s.visual_type.value == 'pdf_image' and s.visual_source_paths)}"
            f"/{len(g.scenes)} 配到圖"
        ),
    )
    _check_budget(start, max_hours)

    pl.stage(
        "合成語音",
        lambda: synthesize_audio.run(
            project_root=p["root"], db=db, config=cfg, resume=True,
            max_retries=int(cfg.get("runtime", {}).get("scene_retry_max", 2)),
        ),
        summary=lambda g: f"{sum(1 for s in g.scenes if s.audio_path)}/{len(g.scenes)} scenes",
    )
    _check_budget(start, max_hours)

    # TTS (coqui-tts) and SDXL share this process and the same GPU. Drop the
    # TTS-side CUDA allocations / Python refs before loading SDXL — otherwise
    # the SDXL UNet forward pass occasionally deadlocks on layer_norm with
    # GPU 0% util, single-thread CPU pegged.
    _release_gpu()

    pl.stage(
        "產生視覺",
        lambda: render_visuals.run(
            project_root=p["root"], db=db, config=cfg, resume=True,
        ),
        summary=lambda g: f"{sum(1 for s in g.scenes if s.visual_asset_paths)}/{len(g.scenes)} scenes",
    )
    _check_budget(start, max_hours)

    pl.stage(
        "建立字幕",
        lambda: build_subtitles.run(project_root=p["root"], db=db, config=cfg),
        summary=lambda g: f"{len(g.scenes)} scenes",
    )
    _check_budget(start, max_hours)

    if not will_render:
        console.print()
        if skip_render:
            console.print("[yellow]skip-render: 在 subtitles 後停止[/]")
        else:
            console.print(
                f"[yellow]ffmpeg 不在 PATH,跳過後續 render 步驟[/]\n"
                f"  裝好 ffmpeg 後可只跑後段: "
                f"paperreel render --project {p['root']}"
            )
        return

    pl.stage(
        "Render scene segments",
        lambda: render_segments.run(
            project_root=p["root"], db=db, config=cfg, resume=True,
        ),
        summary=lambda r: f"{len(r[1].scenes)} scenes",
    )
    _check_budget(start, max_hours)

    final_path = pl.stage(
        "Concat 最終影片",
        lambda: concat_final.run(project_root=p["root"], db=db, config=cfg),
        summary=lambda f: Path(f).name,
    )
    _check_budget(start, max_hours)

    pl.stage(
        "Quality check",
        lambda: quality_check.run(project_root=p["root"], db=db, config=cfg),
        summary=lambda r: (
            f"{r.rendered_scene_count}/{r.scene_count} scenes, "
            f"{r.final_duration_sec/60:.1f} 分鐘, {len(r.issues)} 問題"
        ),
    )

    console.print()
    console.print(f"[green bold]✓ 完成![/]  輸出: {final_path}")


@app.command()
def init(
    project_dir: str = typer.Argument(..., help="Path to create / re-use as project root"),
    overlay: Optional[str] = typer.Option(None, "--config", "-c",
                                          help="Optional config overlay (bundled name like 'bigvram', or filesystem path)"),
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


@app.command(name="match-visuals")
def match_visuals(
    project: str = typer.Option(..., "--project", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Assign extracted PDF figures to matching scenes."""
    p, cfg, db, _ = _ensure_project(project, overlay=config)
    g = match_pdf_visuals.run(project_root=p["root"], db=db, config=cfg)
    # Count matches on visual_source_paths (the upstream figure input) —
    # visual_asset_paths only gets populated later by render_visuals, so
    # running this command standalone would print 0/N otherwise.
    matched = sum(1 for s in g.scenes
                  if s.visual_type.value == "pdf_image" and s.visual_source_paths)
    console.print(f"[green]✓[/green] matched {matched}/{len(g.scenes)} "
                  f"scenes to PDF figures")


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
    order = ["ingest", "plan", "script", "scenes", "match_visuals",
             "audio", "visuals", "subtitles", "segments", "concat",
             "quality"]
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
    resolved = s.get("error_count_resolved", 0)
    if resolved > 0:
        console.print(
            f"errors: {s['error_count']} active "
            f"([dim]{resolved} resolved by a later successful stage run[/dim])")
    else:
        console.print(f"errors: {s['error_count']}")


@app.command(name="retry-failed")
def retry_failed(
    project: str = typer.Option(..., "--project", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Reset failed scenes -> pending so the next audio/visual/render pass retries them."""
    p, cfg, db, _ = _ensure_project(project, overlay=config)
    n = db.reset_failed_scenes()
    console.print(f"[green]✓[/green] reset {n} failed scenes to pending")


# Subcommand names that the bare-PDF wrapper must NOT rewrite. Keep in
# sync with @app.command(...) registrations above.
_KNOWN_SUBCOMMANDS = frozenset({
    "run", "init", "ingest", "plan", "script", "scenes", "match-visuals",
    "audio", "visuals", "subtitles", "render", "status", "retry-failed",
})

# Hold a reference to the real Typer instance so the wrapper below can
# still invoke it after we rebind the public ``app`` symbol.
_typer_app = app


def main_entry() -> None:
    """Entry point for the ``paperreel`` console script.

    Lets ``paperreel <pdf> [opts]`` work as a shorthand for
    ``paperreel run <pdf> [opts]``. Typer's callback can't combine a
    positional argument with subcommand routing (the positional eats the
    subcommand name before Click can dispatch), so we keep ``run`` as a
    real subcommand and rewrite ``sys.argv`` here when the user omitted
    it. Constraint: the PDF must come before any flags in the bare form;
    users who flag-first should write ``paperreel run …`` explicitly.
    """
    import sys
    args = sys.argv[1:]
    if args and not args[0].startswith("-") and args[0] not in _KNOWN_SUBCOMMANDS:
        sys.argv = [sys.argv[0], "run", *args]
    _typer_app()


# Backwards-compat: existing installations whose entry script reads
# ``from paperreel.cli import app; app()`` should also go through the
# wrapper — otherwise the bare-PDF form silently regresses for anyone
# who hasn't reinstalled yet.
app = main_entry  # type: ignore[assignment]


if __name__ == "__main__":
    main_entry()
