"""Stage 3 — outline + per-page text -> script.json (list of ScriptScene)."""
from __future__ import annotations

from pathlib import Path

from ..hashing import hash_inputs
from ..io_utils import atomic_write_json, read_json
from ..models import ChunkedSources, LessonOutline, Script, ScriptScene
from ..providers.llm_base import make_llm_provider
from ..state import StateDB
from ..utils.duration import split_chars_per_scene


def paths_for(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    return {
        "chunked": root / "intermediate" / "chunked_sources.json",
        "outline": root / "intermediate" / "lesson_outline.json",
        "duration": root / "intermediate" / "duration_plan.json",
        "script": root / "intermediate" / "script.json",
    }


def run(*, project_root: str | Path, db: StateDB, config: dict,
        force: bool = False) -> Script:
    p = paths_for(project_root)
    outline = LessonOutline.model_validate(read_json(p["outline"]))
    sources = ChunkedSources.model_validate(read_json(p["chunked"]))
    duration_plan = read_json(p["duration"])

    llm_cfg = config.get("llm", {})
    dur_cfg = config.get("duration", {})

    chars_per_scene, _scene_count = split_chars_per_scene(
        duration_plan["target_seconds"],
        scene_min_sec=float(dur_cfg.get("scene_seconds_min", 30)),
        scene_max_sec=float(dur_cfg.get("scene_seconds_max", 90)),
        chars_per_minute=float(dur_cfg.get("speech_chars_per_minute", 240)),
    )

    input_hash = hash_inputs(
        "script_v1", sources.pdf_sha256, outline.model_dump(mode="json"),
        chars_per_scene, llm_cfg.get("provider"), llm_cfg.get("model"),
        llm_cfg.get("forbid_verbatim_copy", True),
    )
    outputs = [str(p["script"])]
    if not force and db.stage_is_done("script", input_hash, outputs):
        return Script.model_validate(read_json(p["script"]))

    db.start_stage("script", input_hash)
    try:
        page_text = {pg.page: pg.text for pg in sources.pages}
        provider = make_llm_provider(llm_cfg)
        all_scenes: list[ScriptScene] = []
        for ch in outline.chapters:
            scene_dicts = provider.write_chapter_script(
                ch.model_dump(mode="json"),
                page_text,
                chars_per_scene=chars_per_scene,
                forbid_verbatim=bool(llm_cfg.get("forbid_verbatim_copy", True)),
            )
            for sd in scene_dicts:
                all_scenes.append(ScriptScene.model_validate(sd))

        script = Script(
            project=outline.project,
            total_estimated_minutes=sum(
                s.estimated_duration_sec for s in all_scenes
            ) / 60.0,
            scenes=all_scenes,
        )
        atomic_write_json(p["script"], script.model_dump(mode="json"))
        db.register_artifact(p["script"], stage="script", media_type="application/json")
        db.finish_stage("script", outputs)
        return script
    except Exception as e:
        db.fail_stage("script", repr(e))
        db.log_error("script", str(e))
        raise
