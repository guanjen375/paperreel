"""Stage 2 — chunk summaries -> lesson_outline.json (+ duration plan)."""
from __future__ import annotations

from pathlib import Path

from ..hashing import hash_inputs
from ..io_utils import atomic_write_json, read_json
from ..models import ChunkedSources, LessonOutline
from ..providers.llm_base import make_llm_provider
from ..state import StateDB
from ..utils.duration import estimate_target_minutes


def paths_for(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    return {
        "chunked": root / "intermediate" / "chunked_sources.json",
        "outline": root / "intermediate" / "lesson_outline.json",
        "duration": root / "intermediate" / "duration_plan.json",
    }


def run(*, project_root: str | Path, project_name: str, db: StateDB, config: dict,
        target_minutes: float | str = "auto", force: bool = False) -> LessonOutline:
    p = paths_for(project_root)
    sources = ChunkedSources.model_validate(read_json(p["chunked"]))

    llm_cfg = config.get("llm", {})
    dur_cfg = config.get("duration", {})

    plan = estimate_target_minutes(
        cjk_char_count=sources.cjk_char_count,
        page_count=sources.page_count,
        heading_count=sources.heading_count,
        auto_chars_per_minute=float(dur_cfg.get("auto_chars_per_minute", 2200)),
        auto_minutes_min=float(dur_cfg.get("auto_minutes_min", 12)),
        auto_minutes_max=float(dur_cfg.get("auto_minutes_max", 120)),
        user_target=target_minutes,
    )

    input_hash = hash_inputs(
        "plan_v1", sources.pdf_sha256, plan.target_minutes,
        llm_cfg.get("provider"), llm_cfg.get("model"),
    )
    outputs = [str(p["outline"]), str(p["duration"])]
    if not force and db.stage_is_done("plan", input_hash, outputs):
        return LessonOutline.model_validate(read_json(p["outline"]))

    db.start_stage("plan", input_hash)
    try:
        atomic_write_json(p["duration"], {
            "target_minutes": plan.target_minutes,
            "target_seconds": plan.target_seconds,
            "chars_per_minute": plan.chars_per_minute,
            "rationale": plan.rationale,
        })

        provider = make_llm_provider(llm_cfg)
        chunk_summaries: list[dict] = []
        for ch in sources.chunks:
            cs = provider.chunk_summarize(
                ch.text,
                page_range=(ch.start_page, ch.end_page),
                target_chars=400,
            )
            cs.setdefault("page_range", [ch.start_page, ch.end_page])
            chunk_summaries.append(cs)

        outline_dict = provider.build_outline(
            chunk_summaries,
            target_minutes=plan.target_minutes,
            project=project_name,
        )
        outline = LessonOutline.model_validate(outline_dict)
        atomic_write_json(p["outline"], outline.model_dump(mode="json"))
        db.register_artifact(p["outline"], stage="plan", media_type="application/json")
        db.register_artifact(p["duration"], stage="plan", media_type="application/json")
        db.finish_stage("plan", outputs)
        return outline
    except Exception as e:
        db.fail_stage("plan", repr(e))
        db.log_error("plan", str(e))
        raise
