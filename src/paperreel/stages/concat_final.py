"""Stage 9 — concat per-scene MP4 segments into final.mp4."""
from __future__ import annotations

from pathlib import Path

from ..hashing import hash_inputs
from ..io_utils import read_json
from ..models import RenderPlan
from ..renderers.ffmpeg_renderer import (concat_segments, have_ffmpeg,
                                         FfmpegMissing, probe_duration_seconds)
from ..state import StateDB


def paths_for(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    return {
        "render_plan": root / "intermediate" / "render_plan.json",
        "final": root / "outputs" / "final.mp4",
    }


def run(*, project_root: str | Path, db: StateDB, config: dict) -> str:
    p = paths_for(project_root)
    plan = RenderPlan.model_validate(read_json(p["render_plan"]))
    rcfg = config.get("renderer", {})
    ffbin = rcfg.get("ffmpeg_binary", "ffmpeg")
    if not have_ffmpeg(ffbin):
        raise FfmpegMissing(f"ffmpeg binary '{ffbin}' not found on PATH")

    missing: list[str] = []
    for e in plan.entries:
        if not Path(e.output_path).exists():
            missing.append(e.scene_id)
    if missing:
        msg = f"cannot concat: missing segments for scenes: {missing}"
        db.log_error("concat", msg)
        raise FileNotFoundError(msg)

    input_hash = hash_inputs("concat_v1",
                             [e.scene_id for e in plan.entries],
                             [e.output_path for e in plan.entries])
    db.start_stage("concat", input_hash)
    try:
        concat_segments([e.output_path for e in plan.entries], p["final"],
                        ffmpeg_binary=ffbin)
        duration = probe_duration_seconds(p["final"])
        db.register_artifact(p["final"], stage="concat",
                             media_type="video/mp4", duration_sec=duration)
        db.finish_stage("concat", [str(p["final"])])
        return str(p["final"])
    except Exception as e:
        db.fail_stage("concat", repr(e))
        db.log_error("concat", str(e))
        raise
