"""Stage 4 — script.json -> scene_graph.json + scene rows in DB.

This is where the script (LLM output) is promoted to the canonical Scene
objects every downstream stage works on. Each Scene gets a stable
`input_hash` derived from (narration, visual_type, source_pages, voice
config) so a later stage can detect that re-rendering is unnecessary.
"""
from __future__ import annotations

from pathlib import Path

from ..hashing import hash_inputs
from ..io_utils import atomic_write_json, read_json
from ..models import Scene, SceneGraph, SceneStatus, Script, VisualType
from ..state import StateDB
from ..utils.duration import should_insert_recap
from ..utils.source_mapping import format_source_ref


def paths_for(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    return {
        "script": root / "intermediate" / "script.json",
        "duration": root / "intermediate" / "duration_plan.json",
        "scene_graph": root / "intermediate" / "scene_graph.json",
    }


def _scene_input_hash(narration: str, vt: VisualType, source_pages: list[int],
                      voice: str, sample_rate: int) -> str:
    return hash_inputs("scene_v1", narration, vt.value,
                       sorted(source_pages), voice, sample_rate)


def _is_sketchbook(config: dict) -> bool:
    style = (config.get("project") or {}).get("style") or "default"
    return str(style).lower() in ("sketchbook", "document_explainer")


def _anchor_source_paths(anchor) -> list[str]:
    if anchor is None:
        return []
    for attr in ("crop_path", "image_path", "page_render_path"):
        val = getattr(anchor, attr, None)
        if val:
            return [str(val)]
    return []


def run(*, project_root: str | Path, project_name: str,
        pdf_name: str, db: StateDB, config: dict,
        force: bool = False) -> SceneGraph:
    p = paths_for(project_root)
    script = Script.model_validate(read_json(p["script"]))
    duration_plan = read_json(p["duration"])
    tts_cfg = config.get("tts", {})
    voice = str(tts_cfg.get("voice", "default"))
    sample_rate = int(tts_cfg.get("sample_rate_hz", 48000))
    sketchbook = _is_sketchbook(config)

    input_hash = hash_inputs(
        "scenegraph_v3_visual_anchor",
        script.model_dump(mode="json"),
        voice, sample_rate, sketchbook,
    )
    outputs = [str(p["scene_graph"])]
    if not force and db.stage_is_done("scenes", input_hash, outputs):
        return SceneGraph.model_validate(read_json(p["scene_graph"]))

    db.start_stage("scenes", input_hash)
    try:
        # Promote each ScriptScene + insert recap cards at intervals.
        scenes: list[Scene] = []
        recap_every_min = float(config.get("duration", {}).get("recap_every_minutes", 8))
        avg_scene_sec = max(20.0, duration_plan["target_seconds"] /
                            max(1, len(script.scenes)))
        for i, ss in enumerate(script.scenes):
            vt = VisualType(ss.visual_type) if isinstance(ss.visual_type, str) else ss.visual_type
            source_refs = ss.source_refs or [format_source_ref(pdf_name, ss.source_pages)]
            h = _scene_input_hash(ss.narration_text_zh_tw, vt, ss.source_pages,
                                  voice, sample_rate)
            scenes.append(Scene(
                scene_id=ss.scene_id,
                chapter_id=ss.chapter_id,
                title=ss.title,
                source_pages=ss.source_pages,
                source_refs=source_refs,
                narration_text_zh_tw=ss.narration_text_zh_tw,
                visual_prompt=ss.visual_hint or ss.title,
                visual_type=vt,
                on_screen_text=ss.on_screen_text,
                estimated_duration_sec=ss.estimated_duration_sec,
                input_hash=h,
                status=SceneStatus.pending,
                visual_source_paths=_anchor_source_paths(ss.visual_anchor),
                # Carry sketchbook fields through verbatim. Default mode
                # leaves these empty, which is what the renderer expects.
                scene_kind=ss.scene_kind,
                facts=list(ss.facts),
                evidence_spans=list(ss.evidence_spans),
                layout_payload=dict(ss.layout_payload),
                importance=ss.importance,
                visual_anchor=ss.visual_anchor,
                screen_plan=ss.screen_plan,
            ))
            # Recap insertion only applies to default mode — sketchbook
            # already places its own recap_card scene at the end.
            if sketchbook:
                continue
            if should_insert_recap(
                i + 1, len(script.scenes),
                recap_every_minutes=recap_every_min,
                avg_scene_seconds=avg_scene_sec,
            ):
                recap_title = f"{ss.chapter_id} 重點回顧"
                narration = f"我們先停下來回顧本段重點：{ss.title}。"
                h2 = _scene_input_hash(narration, VisualType.recap, ss.source_pages,
                                       voice, sample_rate)
                scenes.append(Scene(
                    scene_id=f"{ss.scene_id}_recap",
                    chapter_id=ss.chapter_id,
                    title=recap_title,
                    source_pages=ss.source_pages,
                    source_refs=source_refs,
                    narration_text_zh_tw=narration,
                    visual_prompt="recap",
                    visual_type=VisualType.recap,
                    on_screen_text=ss.title,
                    estimated_duration_sec=max(20.0, avg_scene_sec / 2.0),
                    input_hash=h2,
                    status=SceneStatus.pending,
                ))

        graph = SceneGraph(
            project=project_name,
            target_minutes=float(duration_plan["target_minutes"]),
            scenes=scenes,
        )
        atomic_write_json(p["scene_graph"], graph.model_dump(mode="json"))
        db.register_artifact(p["scene_graph"], stage="scenes",
                             media_type="application/json")
        for s in scenes:
            db.upsert_scene(
                s.scene_id, s.chapter_id, s.status.value,
                s.input_hash, s.model_dump(mode="json"),
            )
        db.finish_stage("scenes", outputs)
        return graph
    except Exception as e:
        db.fail_stage("scenes", repr(e))
        db.log_error("scenes", str(e))
        raise
