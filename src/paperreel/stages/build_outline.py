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
from ..utils.scene_budget import auto_target_minutes, resolve_target


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
    profile: DocProfile | None = None

    if sketchbook_mode:
        # Classify before duration resolution so target_minutes=auto can
        # account for document family and source length. Explicit
        # --target-minutes still wins.
        profile = doc_classify.classify(sources)
        if target_minutes is None or str(target_minutes).lower() == "auto":
            auto_minutes, auto_rationale = auto_target_minutes(
                page_count=sources.page_count,
                cjk_char_count=sources.cjk_char_count,
                doc_kind=profile.doc_kind.value,
                depth=depth,
            )
            sb_target = resolve_target(depth=depth, target_minutes=auto_minutes)
            plan_rationale = auto_rationale
        else:
            sb_target = resolve_target(depth=depth, target_minutes=target_minutes)
            plan_rationale = sb_target.rationale
        plan_target_minutes = sb_target.target_seconds / 60.0
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

        if sketchbook_mode and profile is not None:
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

        try:
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
        except Exception as e:
            if not sketchbook_mode:
                raise
            db.log_error(
                "plan",
                f"LLM outline skipped, using deterministic explainer outline: {e!r}",
            )
            outline = _heuristic_outline(
                sources=sources,
                target_minutes=plan_target_minutes,
                project=project_name,
            )
        atomic_write_json(p["outline"], outline.model_dump(mode="json"))
        db.register_artifact(p["outline"], stage="plan", media_type="application/json")
        db.register_artifact(p["duration"], stage="plan", media_type="application/json")
        db.finish_stage("plan", outputs)
        return outline
    except Exception as e:
        db.fail_stage("plan", repr(e))
        db.log_error("plan", str(e))
        raise


def _get_attr(obj, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _heuristic_outline(*, sources: ChunkedSources, target_minutes: float,
                       project: str) -> LessonOutline:
    """Build a deterministic outline when no local LLM is available.

    The sketchbook script builder does the factual selection later, so
    this outline only needs stable chapter page ranges and short titles.
    """
    chunks = list(sources.chunks)
    if not chunks:
        joined = "\n".join(pg.text for pg in sources.pages)
        chunks = [{
            "start_page": 1,
            "end_page": max(1, sources.page_count),
            "text": joined,
            "headings": [],
        }]

    # Keep enough chapters to cover the source, but avoid forcing users
    # through a chapter per page for long PDFs. Factual expansion later
    # handles longer target durations.
    desired = max(1, min(len(chunks), round(max(1.0, target_minutes) / 1.5)))
    bucket_size = max(1, (len(chunks) + desired - 1) // desired)
    chapters = []
    for idx in range(0, len(chunks), bucket_size):
        bucket = chunks[idx: idx + bucket_size]
        start_page = int(_get_attr(bucket[0], "start_page", 1) or 1)
        end_page = int(_get_attr(bucket[-1], "end_page", start_page) or start_page)
        text = "\n".join(str(_get_attr(ch, "text", "") or "") for ch in bucket)
        headings = []
        for ch in bucket:
            headings.extend(_get_attr(ch, "headings", []) or [])
        title = _outline_title(text, headings, fallback=f"第 {len(chapters) + 1} 段重點")
        pages = list(range(start_page, end_page + 1)) or [1]
        key_points = _outline_key_points(text)
        chapters.append({
            "chapter_id": f"ch_{len(chapters) + 1:03d}",
            "title": title,
            "source_pages": pages,
            "target_minutes": round(float(target_minutes) / max(1, desired), 2),
            "key_points": key_points,
            "recap": False,
        })
    return LessonOutline.model_validate({
        "project": project,
        "language": "zh-TW",
        "target_minutes": float(target_minutes),
        "rationale": "deterministic explainer outline from PDF chunks",
        "chapters": chapters,
    })


def _outline_title(text: str, headings: list[str], *, fallback: str) -> str:
    for h in headings:
        h = str(h).strip()
        if h:
            return h[:40]
    for line in text.splitlines():
        line = line.strip()
        if 4 <= len(line) <= 60:
            return line[:40]
    return fallback


def _outline_key_points(text: str, *, max_items: int = 3) -> list[str]:
    out: list[str] = []
    for line in text.replace("。", "。\n").splitlines():
        line = line.strip()
        if len(line) < 8:
            continue
        out.append(line[:80])
        if len(out) >= max_items:
            break
    return out


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
