"""Stage 6 — render the per-scene visual layer (PNGs).

For each scene:
- If `visual_type == generated_image` and no asset is on disk yet, call
  the local SDXL provider (image.provider=sdxl) using `visual_prompt`.
  Falls back to the Pillow card renderer if SDXL fails — the pipeline
  must not stall on one bad image.
- All other visual types are rendered by `CardRenderer` (deterministic,
  CPU, no GPU needed).
"""
from __future__ import annotations

from pathlib import Path

from ..hashing import hash_inputs
from ..io_utils import atomic_write_json, read_json
from ..models import Scene, SceneGraph, SceneStatus, VisualType
from ..providers.image_base import ImageProvider, make_image_provider
from ..renderers.card_renderer import CardRenderer
from ..state import StateDB


def paths_for(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    return {
        "scene_graph": root / "intermediate" / "scene_graph.json",
        "visuals_dir": root / "assets" / "visuals",
        "generated_dir": root / "assets" / "generated",
    }


def _generate_sdxl_asset(scene: Scene, out_path: Path,
                         provider: ImageProvider,
                         resolution: tuple[int, int]) -> str | None:
    """Generate one SDXL image. Return path on success, None on failure
    (caller falls back to a card)."""
    prompt = (scene.visual_prompt or scene.title or "").strip()
    if not prompt:
        return None
    try:
        return provider.generate(prompt, out_path,
                                 width=resolution[0], height=resolution[1])
    except Exception:
        return None


def run(*, project_root: str | Path, db: StateDB, config: dict,
        resume: bool = True) -> SceneGraph:
    p = paths_for(project_root)
    graph = SceneGraph.model_validate(read_json(p["scene_graph"]))
    rcfg = config.get("renderer", {})
    icfg = config.get("image", {})
    res = tuple(rcfg.get("resolution", [1920, 1080]))
    renderer = CardRenderer(
        resolution=(int(res[0]), int(res[1])),
        background=rcfg.get("background_color", "#0F172A"),
        foreground=rcfg.get("text_color", "#F8FAFC"),
        accent=rcfg.get("accent_color", "#22D3EE"),
        font_path=rcfg.get("font_path"),
    )
    input_hash = hash_inputs("visuals_v1", rcfg, icfg)
    db.start_stage("visuals", input_hash)

    # Lazily created on first generated_image scene; reused across the whole
    # stage so the SDXL pipeline (~7 GB) is loaded once, not per-scene.
    image_provider: ImageProvider | None = None
    image_provider_failed = False  # so we don't retry the load every scene

    def _drop_gpu_scratch() -> None:
        # Called after every SDXL render. SDXL's per-step transient tensors
        # fragment the CUDA caching allocator; without this we eventually hit
        # a state where layer_norm hangs (single-thread CPU, GPU 0% util) on
        # a later scene's forward pass.
        import gc
        gc.collect()
        try:
            import torch  # noqa: WPS433
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except ImportError:
            pass

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

        # 1) For generated_image scenes, run SDXL into assets/generated/<id>.png.
        #    The result is consumed by the card renderer (inset image), then
        #    visual_asset_paths is rewritten to point at the final card so
        #    later stages (segments / quality) read the renderable PNG.
        sdxl_src: str | None = None
        if sc.visual_type == VisualType.generated_image and not sc.visual_asset_paths:
            if image_provider is None and not image_provider_failed:
                try:
                    image_provider = make_image_provider(icfg)
                except Exception:
                    image_provider_failed = True
            gen_path = p["generated_dir"] / f"{sc.scene_id}.png"
            sdxl_src = (
                _generate_sdxl_asset(sc, gen_path, image_provider,
                                     (int(res[0]), int(res[1])))
                if image_provider is not None
                else None
            )
            _drop_gpu_scratch()
            if sdxl_src:
                sc = sc.model_copy(update={"visual_asset_paths": [sdxl_src]})
                db.register_artifact(Path(sdxl_src), stage="visuals",
                                     scene_id=sc.scene_id,
                                     media_type="image/png",
                                     provenance={"role": "sdxl_source",
                                                 "model": icfg.get("model")})
            else:
                # Downgrade so card renderer takes over.
                sc = sc.model_copy(update={"visual_type": VisualType.bullet_card})

        # 2) PDF-image visuals require the asset to exist on disk.
        if sc.visual_type == VisualType.pdf_image:
            if not sc.visual_asset_paths or not Path(sc.visual_asset_paths[0]).exists():
                sc = sc.model_copy(update={"visual_type": VisualType.bullet_card})

        # 3) Render the final card via CardRenderer.
        try:
            renderer.render_scene(sc, out_path)
        except Exception as e:
            try:
                fb = sc.model_copy(update={"visual_type": VisualType.bullet_card,
                                           "visual_asset_paths": []})
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

        # visual_asset_paths[0] must be the final renderable card; SDXL
        # source stays on disk under assets/generated/ but isn't tracked
        # here (downstream readers always use [0]).
        new_status = (SceneStatus.visual_done
                      if sc.status != SceneStatus.failed else sc.status)
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
