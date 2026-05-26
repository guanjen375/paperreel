"""Stage 3 — outline + per-page text -> script.json (list of ScriptScene).

Default style runs the existing LLM-driven chapter-by-chapter writer.
When ``project.style`` is sketchbook / document_explainer we delegate
to :mod:`build_sketchbook` instead, then validate every scene against
the ingested PDF before allowing the pipeline to advance.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..hashing import hash_inputs
from ..io_utils import atomic_write_json, read_json
from ..models import (ChunkedSources, DocProfile, LessonOutline, Script,
                       ScriptScene, VisualType)
from ..providers.llm_base import make_llm_provider
from ..state import StateDB
from ..utils import grounding
from ..utils.duration import split_chars_per_scene
from ..utils.scene_budget import resolve_target
from . import build_sketchbook


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
        "doc_profile": root / "intermediate" / "doc_profile.json",
        "plan_report": root / "intermediate" / "sketchbook_plan.json",
    }


def _is_sketchbook(config: dict) -> bool:
    style = (config.get("project") or {}).get("style") or "default"
    return str(style).lower() in ("sketchbook", "document_explainer")


def run(*, project_root: str | Path, db: StateDB, config: dict,
        force: bool = False) -> Script:
    p = paths_for(project_root)
    outline = LessonOutline.model_validate(read_json(p["outline"]))
    sources = ChunkedSources.model_validate(read_json(p["chunked"]))
    duration_plan = read_json(p["duration"])

    llm_cfg = config.get("llm", {})
    dur_cfg = config.get("duration", {})

    if _is_sketchbook(config):
        return _run_sketchbook(
            project_root=project_root, db=db, config=config,
            outline=outline, sources=sources,
            duration_plan=duration_plan, force=force,
        )

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


# ---------- sketchbook branch ----------

def _run_sketchbook(*, project_root: str | Path, db: StateDB, config: dict,
                    outline: LessonOutline, sources: ChunkedSources,
                    duration_plan: dict, force: bool) -> Script:
    p = paths_for(project_root)
    llm_cfg = config.get("llm", {})
    de_cfg = config.get("doc_explainer", {}) or {}
    grounding_cfg = de_cfg.get("grounding", {}) or {}
    cards_cfg = de_cfg.get("cards", {}) or {}
    project_cfg = config.get("project") or {}
    depth = str(project_cfg.get("depth") or duration_plan.get("depth")
                or "standard").lower()

    if p["doc_profile"].exists():
        profile = DocProfile.model_validate(read_json(p["doc_profile"]))
    else:
        # Fallback when plan stage was an older run that didn't write a
        # profile — classify on the fly so we still pick the right
        # storyboard.
        from ..utils import doc_classify  # local import to avoid cycle
        profile = doc_classify.classify(sources)

    duration = resolve_target(
        depth=depth,
        target_minutes=duration_plan.get("target_minutes"),
    )

    input_hash = hash_inputs(
        "sketchbook_script_v2_visual_first",
        sources.pdf_sha256,
        outline.model_dump(mode="json"),
        profile.model_dump(mode="json"),
        duration.target_seconds, duration.depth,
        llm_cfg.get("provider"), llm_cfg.get("model"),
        cards_cfg, grounding_cfg,
    )
    outputs = [str(p["script"]), str(p["plan_report"])]
    if not force and db.stage_is_done("script", input_hash, outputs):
        return Script.model_validate(read_json(p["script"]))

    db.start_stage("script", input_hash)
    try:
        provider = None
        try:
            provider = make_llm_provider(llm_cfg)
        except Exception as e:
            # LLM is best-effort polish in sketchbook mode; heuristic
            # narration is already grounded. Log and continue.
            db.log_error("script",
                         f"LLM polish skipped, falling back to heuristic narration: {e!r}")

        scenes, plan_report = build_sketchbook.build_sketchbook_scenes(
            sources=sources,
            outline=outline,
            profile=profile,
            duration=duration,
            cards_cfg=cards_cfg,
            provider=provider,
            use_llm_polish=provider is not None,
        )

        # Grounding validation. One repair attempt: drop scenes that
        # can't be grounded rather than emit an unbacked claim.
        page_text = {pg.page: pg.text for pg in sources.pages}
        min_ratio = float(grounding_cfg.get("quote_match_min_ratio", 0.55))
        require = bool(grounding_cfg.get("require_evidence_for_facts", True))
        max_repairs = int(grounding_cfg.get("max_repair_attempts", 1))

        issues = grounding.validate_scenes(
            scenes, page_text=page_text,
            min_quote_ratio=min_ratio,
            require_evidence_for_facts=require,
        )
        repaired_round = 0
        while issues and repaired_round < max_repairs:
            scenes = _repair_scenes(scenes, issues)
            repaired_round += 1
            issues = grounding.validate_scenes(
                scenes, page_text=page_text,
                min_quote_ratio=min_ratio,
                require_evidence_for_facts=require,
            )
        if issues:
            # Hard fail with a single clear log entry so the user knows
            # exactly which scene + quote couldn't be grounded.
            msg = "; ".join(f"{i.scene_id}:{i.code}:{i.message[:80]}"
                            for i in issues[:6])
            raise grounding.GroundingError(issues) from None

        _validate_unique_scene_ids(scenes)

        script = Script(
            project=outline.project,
            total_estimated_minutes=sum(
                s.estimated_duration_sec for s in scenes
            ) / 60.0,
            scenes=scenes,
        )
        atomic_write_json(p["script"], script.model_dump(mode="json"))
        atomic_write_json(p["plan_report"], plan_report)
        db.register_artifact(p["script"], stage="script",
                             media_type="application/json")
        db.register_artifact(p["plan_report"], stage="script",
                             media_type="application/json")
        db.finish_stage("script", outputs)
        return script
    except Exception as e:
        db.fail_stage("script", repr(e))
        db.log_error("script", str(e))
        raise


def _repair_scenes(scenes: list[ScriptScene],
                   issues: list["grounding.GroundingIssue"]
                   ) -> list[ScriptScene]:
    """Attempt one round of automatic repair: drop unfixable
    evidence / facts so the remaining scene still teaches something
    grounded. Returns the new scene list — scenes that lose all their
    evidence under a factual ``scene_kind`` are removed outright.
    """
    bad_by_scene: dict[str, list["grounding.GroundingIssue"]] = {}
    for i in issues:
        bad_by_scene.setdefault(i.scene_id, []).append(i)
    out: list[ScriptScene] = []
    for sc in scenes:
        scene_issues = bad_by_scene.get(sc.scene_id, [])
        if not scene_issues:
            out.append(sc)
            continue
        evidence = list(sc.evidence_spans)
        facts = list(sc.facts)
        # Build set of bad evidence indices.
        bad_pages = {i.page for i in scene_issues if i.page is not None
                      and i.code in ("bad_page", "quote_mismatch")}
        if bad_pages:
            evidence = [s for s in evidence if s.page not in bad_pages]
            # Update fact references — drop facts whose evidence is gone.
            facts = [f for f in facts
                      if f.evidence_index is None or
                      f.evidence_index < len(evidence)]
        kind = (sc.scene_kind or "").lower()
        if kind in ("deadline_timeline", "penalty_table", "checklist",
                    "risk_warning", "do_dont", "key_number", "source_crop"):
            if not evidence:
                # Without evidence this scene is unsupported — drop it.
                continue
        out.append(sc.model_copy(update={
            "evidence_spans": evidence,
            "facts": facts,
        }))
    return out
