"""Stage 3 — outline + per-page text -> script.json (list of ScriptScene)."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..hashing import hash_inputs
from ..io_utils import atomic_write_json, read_json
from ..models import ChunkedSources, LessonOutline, Script, ScriptScene
from ..providers.llm_base import make_llm_provider
from ..state import StateDB
from ..utils.duration import split_chars_per_scene


# Canonical IDs are filesystem-safe and sort lexicographically. Anything
# the LLM returns is overwritten — it can't be trusted to produce
# globally-unique scene_ids across chapters, because each chapter prompt
# is independent and the model happily reuses the `_sc_001` slot every
# time. Letting that through silently used to clobber audio / visual /
# segment artefacts (all keyed by scene_id) belonging to a different
# chapter.
_CHAPTER_ID_RE = re.compile(r"^ch_\d{3,}$")


def _canonical_chapter_ids(chapters) -> list[str]:
    """Return canonical chapter ids parallel to ``chapters``.

    Preserves outline-supplied ids when they already look canonical
    (``ch_\\d{3,}``) **and** are unique. Falls back to dense
    position-based ids (``ch_001``, ``ch_002``, …) when anything looks
    off — so downstream filenames / DB keys can never collide.
    """
    proposed = [
        ch.chapter_id if _CHAPTER_ID_RE.match(ch.chapter_id or "") else f"ch_{i:03d}"
        for i, ch in enumerate(chapters, start=1)
    ]
    if len(set(proposed)) == len(proposed):
        return proposed
    return [f"ch_{i:03d}" for i in range(1, len(chapters) + 1)]


def _validate_unique_scene_ids(scenes: list[ScriptScene]) -> None:
    """Raise ValueError if any scene_id appears twice. Cheap last-line
    of defence: with our id normalisation this should never trigger,
    but if it ever does we want to halt before writing script.json and
    cascading the duplicate into audio / visual / segment paths."""
    ids = [s.scene_id for s in scenes]
    if len(ids) == len(set(ids)):
        return
    dupes = sorted({i for i, c in Counter(ids).items() if c > 1})
    raise ValueError(
        f"duplicate scene_id detected after normalization: {dupes!r}. "
        "Refusing to write script.json; this would clobber per-scene "
        "audio/visual/segment artefacts that key off scene_id."
    )


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
        "script_v2", sources.pdf_sha256, outline.model_dump(mode="json"),
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
        canonical_chapter_ids = _canonical_chapter_ids(outline.chapters)

        all_scenes: list[ScriptScene] = []
        for ch_idx, ch in enumerate(outline.chapters):
            chap_id = canonical_chapter_ids[ch_idx]
            scene_dicts = provider.write_chapter_script(
                ch.model_dump(mode="json"),
                page_text,
                chars_per_scene=chars_per_scene,
                forbid_verbatim=bool(llm_cfg.get("forbid_verbatim_copy", True)),
            )
            for sc_idx, raw in enumerate(scene_dicts, start=1):
                # Defensive copy so we don't mutate provider-owned dicts.
                sd = dict(raw)
                # Override unconditionally: scene_id / chapter_id from the
                # LLM are never trusted. Even when the model "obeys" the
                # prompt, two chapters can independently produce the same
                # `_sc_001` slot.
                sd["chapter_id"] = chap_id
                sd["scene_id"] = f"{chap_id}_sc_{sc_idx:03d}"
                all_scenes.append(ScriptScene.model_validate(sd))

        _validate_unique_scene_ids(all_scenes)

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
