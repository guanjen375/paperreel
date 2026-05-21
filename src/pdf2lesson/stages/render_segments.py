"""Stage 8 — mux per-scene PNG + WAV into MP4 segments."""
from __future__ import annotations

from pathlib import Path

from ..hashing import hash_inputs
from ..io_utils import atomic_write_json, read_json
from ..models import RenderPlan, RenderPlanEntry, Scene, SceneGraph, SceneStatus
from ..renderers.ffmpeg_renderer import (FfmpegError, FfmpegMissing,
                                         have_ffmpeg, render_still_with_audio)
from ..state import StateDB


def paths_for(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    return {
        "scene_graph": root / "intermediate" / "scene_graph.json",
        "segments_dir": root / "outputs" / "segments",
        "render_plan": root / "intermediate" / "render_plan.json",
    }


def run(*, project_root: str | Path, db: StateDB, config: dict,
        resume: bool = True, max_retries: int = 2) -> tuple[RenderPlan, SceneGraph]:
    p = paths_for(project_root)
    graph = SceneGraph.model_validate(read_json(p["scene_graph"]))
    rcfg = config.get("renderer", {})
    ffbin = rcfg.get("ffmpeg_binary", "ffmpeg")
    res = tuple(rcfg.get("resolution", [1920, 1080]))
    fps = int(rcfg.get("fps", 30))

    if not have_ffmpeg(ffbin):
        raise FfmpegMissing(
            f"ffmpeg binary '{ffbin}' not found on PATH. "
            "Install ffmpeg (e.g. `sudo apt install ffmpeg` or `choco install ffmpeg`) "
            "then retry."
        )

    input_hash = hash_inputs("segments_v1", rcfg, fps, res)
    db.start_stage("segments", input_hash)

    entries: list[RenderPlanEntry] = []
    new_scenes: list[Scene] = []
    failures: list[str] = []
    for sc in graph.scenes:
        out = p["segments_dir"] / f"{sc.scene_id}.mp4"
        if sc.status == SceneStatus.failed or not sc.audio_path or not sc.visual_asset_paths:
            failures.append(sc.scene_id)
            new_scenes.append(sc)
            continue
        duration = sc.actual_duration_sec or sc.estimated_duration_sec
        if resume and sc.rendered_video_path and Path(sc.rendered_video_path).exists():
            entries.append(RenderPlanEntry(
                scene_id=sc.scene_id,
                visual_type=sc.visual_type,
                visual_asset_paths=sc.visual_asset_paths,
                audio_path=sc.audio_path,
                subtitle_path=sc.subtitle_path,
                output_path=sc.rendered_video_path,
                duration_sec=duration,
            ))
            new_scenes.append(sc)
            continue
        attempt = 0
        last_err: str | None = None
        while attempt <= max_retries:
            try:
                render_still_with_audio(
                    sc.visual_asset_paths[0], sc.audio_path, out,
                    duration_sec=duration,
                    fps=fps,
                    resolution=(int(res[0]), int(res[1])),
                    video_codec=rcfg.get("video_codec", "libx264"),
                    audio_codec=rcfg.get("audio_codec", "aac"),
                    pixel_format=rcfg.get("pixel_format", "yuv420p"),
                    ken_burns=bool(rcfg.get("ken_burns", False)),
                    ffmpeg_binary=ffbin,
                )
                last_err = None
                break
            except (FfmpegError, FfmpegMissing, FileNotFoundError) as e:
                last_err = repr(e)
                attempt += 1
                if attempt > max_retries:
                    break
        if last_err is not None:
            sc = sc.model_copy(update={
                "status": SceneStatus.failed,
                "last_error": last_err,
            })
            db.upsert_scene(sc.scene_id, sc.chapter_id, sc.status.value,
                            sc.input_hash, sc.model_dump(mode="json"),
                            last_error=last_err, bump_retry=True)
            db.log_error("segments", last_err, scene_id=sc.scene_id)
            failures.append(sc.scene_id)
            new_scenes.append(sc)
            continue
        sc = sc.model_copy(update={
            "rendered_video_path": str(out),
            "status": SceneStatus.rendered,
            "last_error": None,
        })
        db.register_artifact(out, stage="segments", scene_id=sc.scene_id,
                             media_type="video/mp4", duration_sec=duration)
        db.upsert_scene(sc.scene_id, sc.chapter_id, sc.status.value,
                        sc.input_hash, sc.model_dump(mode="json"))
        new_scenes.append(sc)
        entries.append(RenderPlanEntry(
            scene_id=sc.scene_id,
            visual_type=sc.visual_type,
            visual_asset_paths=sc.visual_asset_paths,
            audio_path=sc.audio_path,
            subtitle_path=sc.subtitle_path,
            output_path=str(out),
            duration_sec=duration,
        ))

    plan = RenderPlan(
        project=graph.project,
        fps=fps,
        resolution=(int(res[0]), int(res[1])),
        final_output=str(Path(project_root) / "outputs" / "final.mp4"),
        entries=entries,
    )
    atomic_write_json(p["render_plan"], plan.model_dump(mode="json"))
    graph = graph.model_copy(update={"scenes": new_scenes})
    atomic_write_json(p["scene_graph"], graph.model_dump(mode="json"))
    db.finish_stage("segments", [str(p["render_plan"])])
    if failures:
        db.log_error("segments", f"scenes failed to render: {failures}")
    return plan, graph
