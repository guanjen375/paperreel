"""Stage 6 — render the per-scene visual layer (PNGs).

Renders one card per scene using Pillow. If a scene's visual_type points at
a PDF image but that image is missing, falls back to teaching_card so the
pipeline never stalls.
"""
from __future__ import annotations

from pathlib import Path

from ..hashing import hash_inputs
from ..io_utils import atomic_write_json, read_json
from ..models import Scene, SceneGraph, SceneStatus, VisualType
from ..renderers.card_renderer import CardRenderer
from ..state import StateDB


def paths_for(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    return {
        "scene_graph": root / "intermediate" / "scene_graph.json",
        "visuals_dir": root / "assets" / "visuals",
    }


def run(*, project_root: str | Path, db: StateDB, config: dict,
        resume: bool = True) -> SceneGraph:
    p = paths_for(project_root)
    graph = SceneGraph.model_validate(read_json(p["scene_graph"]))
    rcfg = config.get("renderer", {})
    res = tuple(rcfg.get("resolution", [1920, 1080]))
    renderer = CardRenderer(
        resolution=(int(res[0]), int(res[1])),
        background=rcfg.get("background_color", "#0F172A"),
        foreground=rcfg.get("text_color", "#F8FAFC"),
        accent=rcfg.get("accent_color", "#22D3EE"),
        font_path=rcfg.get("font_path"),
    )
    input_hash = hash_inputs("visuals_v1", rcfg)
    db.start_stage("visuals", input_hash)

    new_scenes: list[Scene] = []
    failures: list[str] = []

    for sc in graph.scenes:
        out_path = p["visuals_dir"] / f"{sc.scene_id}.png"
        if resume and sc.visual_asset_paths and Path(sc.visual_asset_paths[0]).exists():
            new_scenes.append(sc)
            continue
        if sc.status == SceneStatus.failed:
            new_scenes.append(sc)
            continue
        try:
            # Validate PDF image presence before letting the renderer decide.
            if sc.visual_type in (VisualType.pdf_image, VisualType.generated_image):
                if not sc.visual_asset_paths or not Path(sc.visual_asset_paths[0]).exists():
                    sc = sc.model_copy(update={"visual_type": VisualType.bullet_card})
            renderer.render_scene(sc, out_path)
        except Exception as e:
            # Fallback to a teaching card so we don't lose the scene.
            try:
                fb = sc.model_copy(update={"visual_type": VisualType.bullet_card})
                renderer.render_scene(fb, out_path)
                sc = fb.model_copy(update={
                    "last_error": f"render fallback: {e!r}",
                })
            except Exception as e2:
                sc = sc.model_copy(update={
                    "status": SceneStatus.failed,
                    "last_error": f"render failed: {e2!r}",
                })
                db.log_error("visuals", str(e2), scene_id=sc.scene_id)
                failures.append(sc.scene_id)
                db.upsert_scene(sc.scene_id, sc.chapter_id, sc.status.value,
                                sc.input_hash, sc.model_dump(mode="json"),
                                last_error=sc.last_error, bump_retry=True)
                new_scenes.append(sc)
                continue

        prev = sc.status if sc.status == SceneStatus.audio_done else sc.status
        new_status = SceneStatus.visual_done if prev != SceneStatus.failed else sc.status
        sc = sc.model_copy(update={
            "visual_asset_paths": [str(out_path)],
            "status": new_status,
        })
        db.register_artifact(out_path, stage="visuals", scene_id=sc.scene_id,
                             media_type="image/png")
        db.upsert_scene(sc.scene_id, sc.chapter_id, sc.status.value,
                        sc.input_hash, sc.model_dump(mode="json"))
        new_scenes.append(sc)

    graph = graph.model_copy(update={"scenes": new_scenes})
    atomic_write_json(p["scene_graph"], graph.model_dump(mode="json"))
    db.finish_stage("visuals", [str(p["scene_graph"])])
    if failures:
        db.log_error("visuals", f"visual render failed for scenes: {failures}")
    return graph
