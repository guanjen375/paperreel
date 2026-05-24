"""Stage 2 — chunk summaries -> lesson_outline.json (+ duration plan).

When ``project.style`` is sketchbook / document_explainer we additionally
write ``intermediate/doc_profile.json`` so the script stage can pick the
right storyboard skeleton.
"""
from __future__ import annotations

from pathlib import Path

from ..hashing import hash_inputs
from ..io_utils import atomic_write_json, read_json
from ..models import ChunkedSources, DocKind, DocProfile, LessonOutline
from ..providers.llm_base import make_llm_provider
from ..state import StateDB
from ..utils import doc_classify
from ..utils.duration import estimate_target_minutes
from ..utils.scene_budget import resolve_target


def paths_for(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    return {
        "chunked": root / "intermediate" / "chunked_sources.json",
        "outline": root / "intermediate" / "lesson_outline.json",
        "duration": root / "intermediate" / "duration_plan.json",
        "doc_profile": root / "intermediate" / "doc_profile.json",
    }


def _is_sketchbook(config: dict) -> bool:
    style = (config.get("project") or {}).get("style") or "default"
    return str(style).lower() in ("sketchbook", "document_explainer")


def run(*, project_root: str | Path, project_name: str, db: StateDB, config: dict,
        target_minutes: float | str = "auto", force: bool = False) -> LessonOutline:
    p = paths_for(project_root)
    sources = ChunkedSources.model_validate(read_json(p["chunked"]))

    llm_cfg = config.get("llm", {})
    dur_cfg = config.get("duration", {})
    sketchbook_mode = _is_sketchbook(config)
    project_cfg = config.get("project") or {}
    depth = str(project_cfg.get("depth") or "standard").lower()

    if sketchbook_mode:
        # Sketchbook uses the depth/target_minutes controller, not the
        # CJK-density estimator (those caps target full chapter videos,
        # not 2–10 min explainers).
        sb_target = resolve_target(depth=depth, target_minutes=target_minutes)
        plan_target_minutes = sb_target.target_seconds / 60.0
        plan_rationale = sb_target.rationale
        plan_chars_per_minute = float(dur_cfg.get("speech_chars_per_minute", 240))
    else:
        plan = estimate_target_minutes(
            cjk_char_count=sources.cjk_char_count,
            page_count=sources.page_count,
            heading_count=sources.heading_count,
            auto_chars_per_minute=float(dur_cfg.get("auto_chars_per_minute", 2200)),
            auto_minutes_min=float(dur_cfg.get("auto_minutes_min", 3)),
            auto_minutes_max=float(dur_cfg.get("auto_minutes_max", 120)),
            user_target=target_minutes,
        )
        plan_target_minutes = plan.target_minutes
        plan_rationale = plan.rationale
        plan_chars_per_minute = plan.chars_per_minute

    input_hash = hash_inputs(
        "plan_v2", sources.pdf_sha256, plan_target_minutes,
        llm_cfg.get("provider"), llm_cfg.get("model"),
        sketchbook_mode, depth,
    )
    outputs = [str(p["outline"]), str(p["duration"])]
    if sketchbook_mode:
        outputs.append(str(p["doc_profile"]))
    if not force and db.stage_is_done("plan", input_hash, outputs):
        return LessonOutline.model_validate(read_json(p["outline"]))

    db.start_stage("plan", input_hash)
    try:
        atomic_write_json(p["duration"], {
            "target_minutes": plan_target_minutes,
            "target_seconds": plan_target_minutes * 60.0,
            "chars_per_minute": plan_chars_per_minute,
            "rationale": plan_rationale,
            "depth": depth if sketchbook_mode else None,
            "style": "sketchbook" if sketchbook_mode else "default",
        })

        if sketchbook_mode:
            profile = doc_classify.classify(sources)
            # Optional LLM refinement — only triggered when the config
            # explicitly opts in. Failure here is non-fatal: we already
            # have a heuristic profile.
            de_cfg = config.get("doc_explainer", {}) or {}
            cls_cfg = de_cfg.get("classify", {}) or {}
            if bool(cls_cfg.get("use_llm_refinement")):
                try:
                    provider = make_llm_provider(llm_cfg)
                    profile = _refine_with_llm(provider, sources, profile)
                except Exception as e:
                    db.log_error("plan", f"LLM doc-kind refinement skipped: {e!r}")
            atomic_write_json(p["doc_profile"], profile.model_dump(mode="json"))
            db.register_artifact(p["doc_profile"], stage="plan",
                                 media_type="application/json")

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
            target_minutes=plan_target_minutes,
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


def _refine_with_llm(provider, sources: ChunkedSources,
                     profile: DocProfile) -> DocProfile:
    """Ask the LLM to confirm or override the heuristic doc_kind.

    Optional. Provider must implement ``classify_document`` for this to
    do anything; absence is treated as "no refinement available".
    """
    fn = getattr(provider, "classify_document", None)
    if not callable(fn):
        return profile
    sample = " ".join(p.text for p in sources.pages[:6])[:6000]
    candidate = fn(sample=sample,
                   heuristic=profile.model_dump(mode="json"))
    if not isinstance(candidate, dict):
        return profile
    kind_raw = str(candidate.get("doc_kind") or profile.doc_kind.value).lower()
    try:
        new_kind = DocKind(kind_raw)
    except ValueError:
        return profile
    return profile.model_copy(update={
        "doc_kind": new_kind,
        "rationale": profile.rationale + f" | llm:{new_kind.value}",
        "suggested_storyboard": doc_classify.storyboard_for(new_kind),
    })
