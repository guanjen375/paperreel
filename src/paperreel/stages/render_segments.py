"""Stage 8 — mux per-scene PNG + WAV into MP4 segments.

Resume model: each ``outputs/segments/<scene_id>.mp4`` carries a sidecar
manifest fingerprinted from (audio manifest hash, visual manifest hash,
fps, codec, pixel_format, resolution, ken_burns). When the upstream
audio or visual is invalidated and regenerated, the corresponding
segment is automatically invalidated too — the chain of input_hash
references composes correctly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..hashing import hash_inputs
from ..io_utils import atomic_write_json, read_json
from ..manifest import manifest_matches, read_manifest, sha256_of, write_manifest
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


def _segment_codec_signature(rcfg: dict, fps: int,
                             res: tuple[int, int]) -> dict[str, Any]:
    """Fields that affect the muxed MP4 bytes."""
    return {
        "fps": fps,
        "resolution": list(res),
        "video_codec": rcfg.get("video_codec"),
        "audio_codec": rcfg.get("audio_codec"),
        "pixel_format": rcfg.get("pixel_format"),
        "ken_burns": bool(rcfg.get("ken_burns", False)),
    }


def _upstream_hash(path: str | None) -> str | None:
    """Pull the upstream stage's input_hash off its sidecar manifest;
    fall back to the file's own SHA when no manifest exists (so older
    artefacts produced before the manifest rollout still chain)."""
    if not path:
        return None
    m = read_manifest(path)
    if m and m.get("input_hash"):
        return m["input_hash"]
    return sha256_of(path)


def _segment_inputs(scene: Scene, codec_sig: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "segment_artifact_v2",
        "scene_id": scene.scene_id,
        "audio_input_hash": _upstream_hash(scene.audio_path),
        "visual_input_hash": _upstream_hash(
            scene.visual_asset_paths[0] if scene.visual_asset_paths else None
        ),
        "duration_sec": (scene.actual_duration_sec
                         or scene.estimated_duration_sec),
        "codec": codec_sig,
    }


def _segment_input_hash(scene: Scene, codec_sig: dict[str, Any]) -> str:
    return hash_inputs("segment_artifact_v2",
                       _segment_inputs(scene, codec_sig))


def run(*, project_root: str | Path, db: StateDB, config: dict,
        resume: bool = True, max_retries: int = 2) -> tuple[RenderPlan, SceneGraph]:
    p = paths_for(project_root)
    graph = SceneGraph.model_validate(read_json(p["scene_graph"]))
    rcfg = config.get("renderer", {})
    ffbin = rcfg.get("ffmpeg_binary", "ffmpeg")
    res = tuple(rcfg.get("resolution", [1920, 1080]))
    res_tuple: tuple[int, int] = (int(res[0]), int(res[1]))
    fps = int(rcfg.get("fps", 30))

    if not have_ffmpeg(ffbin):
        raise FfmpegMissing(
            f"ffmpeg binary '{ffbin}' not found on PATH. "
            "Install ffmpeg (e.g. `sudo apt install ffmpeg` or `choco install ffmpeg`) "
            "then retry."
        )

    codec_sig = _segment_codec_signature(rcfg, fps, res_tuple)
    db.start_stage("segments",
                   hash_inputs("segments_stage_v2", codec_sig))

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
        expected_hash = _segment_input_hash(sc, codec_sig)

        # Resume: skip only when the segment's manifest matches the hash
        # we'd produce now. This composes with upstream — if the audio
        # or visual was regenerated, their manifest hash changed, so
        # this scene's expected_hash changes, and the segment rebuilds.
        if resume and manifest_matches(out, expected_hash):
            sc = sc.model_copy(update={
                "rendered_video_path": str(out),
                "status": (SceneStatus.rendered
                           if sc.status != SceneStatus.failed else sc.status),
            })
            entries.append(RenderPlanEntry(
                scene_id=sc.scene_id,
                visual_type=sc.visual_type,
                visual_asset_paths=sc.visual_asset_paths,
                audio_path=sc.audio_path,
                subtitle_path=sc.subtitle_path,
                output_path=str(out),
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
                    resolution=res_tuple,
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
        write_manifest(
            out,
            stage="segments",
            scene_id=sc.scene_id,
            input_hash=expected_hash,
            inputs=_segment_inputs(sc, codec_sig),
            extra={"duration_sec": duration},
        )
        sc = sc.model_copy(update={
            "rendered_video_path": str(out),
            "status": SceneStatus.rendered,
            "last_error": None,
        })
        db.register_artifact(out, stage="segments", scene_id=sc.scene_id,
                             media_type="video/mp4", duration_sec=duration,
                             provenance={"input_hash": expected_hash})
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
        resolution=res_tuple,
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
