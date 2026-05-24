"""Stage 6 — render the per-scene visual layer (PNGs).

For each scene:
- If `visual_type == generated_image` and no asset is on disk yet, call
  the local SDXL provider (image.provider=sdxl) using `visual_prompt`.
  Falls back to the Pillow card renderer if SDXL fails — the pipeline
  must not stall on one bad image.
- All other visual types are rendered by `CardRenderer` (deterministic,
  CPU, no GPU needed).

Resume model mirrors the audio stage: each rendered card has a sidecar
``.manifest.json`` describing the visual_type / prompt / on_screen_text /
renderer config / source-image SHA that produced it. We skip rendering
only when that hash still matches the current config — so changing the
background colour, the font path, the SDXL model, or the embedded PDF
crop all invalidate the cache correctly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..hashing import hash_inputs
from ..io_utils import atomic_write_json, read_json
from ..manifest import manifest_matches, sha256_of, write_manifest
from ..models import Scene, SceneGraph, SceneStatus, VisualType
from ..providers.image_base import ImageProvider, make_image_provider
from ..renderers.card_renderer import CardRenderer
from ..renderers.sketchbook_renderer import SketchbookRenderer
from ..state import StateDB


def paths_for(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    return {
        "scene_graph": root / "intermediate" / "scene_graph.json",
        "visuals_dir": root / "assets" / "visuals",
        "generated_dir": root / "assets" / "generated",
        "crops_dir": root / "assets" / "source_crops",
    }


def _is_sketchbook(config: dict) -> bool:
    style = (config.get("project") or {}).get("style") or "default"
    return str(style).lower() in ("sketchbook", "document_explainer")


def _renderer_config_signature(rcfg: dict) -> dict[str, Any]:
    """Only the renderer fields that actually affect pixel output."""
    return {
        "resolution": list(rcfg.get("resolution", [1920, 1080])),
        "background_color": rcfg.get("background_color"),
        "text_color": rcfg.get("text_color"),
        "accent_color": rcfg.get("accent_color"),
        "font_path": rcfg.get("font_path"),
    }


def _image_config_signature(icfg: dict) -> dict[str, Any]:
    """Only the SDXL fields that actually affect what the model emits.

    Excludes runtime knobs like device/dtype: switching cuda <-> cpu or
    fp16 <-> bf16 should NOT invalidate the cache.
    """
    return {
        "provider": icfg.get("provider"),
        "model": icfg.get("model"),
        "num_inference_steps": icfg.get("num_inference_steps"),
        "guidance_scale": icfg.get("guidance_scale"),
        "negative_prompt": icfg.get("negative_prompt"),
    }


def _visual_inputs(scene: Scene, rcfg: dict, icfg: dict,
                   *,
                   source_asset_path: str | None,
                   source_asset_sha: str | None) -> dict[str, Any]:
    """Fully fingerprint one visual artefact.

    ``source_asset_path`` / ``source_asset_sha`` describe the upstream
    image fed into the card (extracted PDF figure crop, freshly
    generated SDXL image) — both None for pure text cards.
    """
    return {
        "schema": "visual_artifact_v4",
        "scene_id": scene.scene_id,
        "visual_type": scene.visual_type.value,
        "visual_prompt": scene.visual_prompt,
        "on_screen_text": scene.on_screen_text,
        "title": scene.title,
        "narration_len": len(scene.narration_text_zh_tw),
        "source_pages": list(scene.source_pages),
        "source_asset_path": source_asset_path,
        "source_asset_sha256": source_asset_sha,
        "renderer": _renderer_config_signature(rcfg),
        "image": _image_config_signature(icfg)
            if scene.visual_type == VisualType.generated_image else None,
    }


def _visual_input_hash(scene: Scene, rcfg: dict, icfg: dict,
                       *,
                       source_asset_path: str | None,
                       source_asset_sha: str | None) -> str:
    return hash_inputs(
        "visual_artifact_v4",
        _visual_inputs(scene, rcfg, icfg,
                       source_asset_path=source_asset_path,
                       source_asset_sha=source_asset_sha),
    )


def _source_path_for(scene: Scene) -> str | None:
    """Return the upstream source image path for ``scene``, or None.

    Reads :attr:`Scene.visual_source_paths` — the dedicated *input*
    field — never ``visual_asset_paths`` (which holds the renderer's
    *output*). Mixing those two caused a self-nesting bug on second
    runs, where match_visuals saw the prior render's card path and
    skipped, then render_visuals re-embedded that card as its own
    inset.
    """
    if not scene.visual_source_paths:
        return None
    return scene.visual_source_paths[0]


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
    sketchbook = _is_sketchbook(config)
    de_cfg = config.get("doc_explainer", {}) or {}
    allow_generated = bool(de_cfg.get("allow_generated_images", False))
    sketchbook_renderer: SketchbookRenderer | None = None
    if sketchbook:
        sketchbook_renderer = SketchbookRenderer(
            resolution=(int(res[0]), int(res[1])),
            background=rcfg.get("background_color", "#F8FAFB"),
            foreground=rcfg.get("text_color", "#0F172A"),
            accent=rcfg.get("accent_color", "#D97706"),
            font_path=rcfg.get("font_path"),
            cards_cfg=de_cfg.get("cards"),
        )
    db.start_stage(
        "visuals",
        hash_inputs("visuals_stage_v3",
                    _renderer_config_signature(rcfg),
                    _image_config_signature(icfg),
                    sketchbook, allow_generated),
    )

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

        # In sketchbook mode the script writer has already populated
        # scene_kind / layout_payload, so we can render deterministically
        # without an SDXL / PDF crop step (unless scene_kind is
        # source_crop, which we still want to handle below).
        is_sketchbook_scene = bool(sketchbook and sc.scene_kind and
                                    sketchbook_renderer is not None)

        # Forbid SDXL generation when allow_generated_images is off.
        # Downgrade to a sketchbook paragraph card or bullet card so the
        # pipeline keeps going instead of emitting a stale image.
        if sketchbook and sc.visual_type == VisualType.generated_image and not allow_generated:
            sc = sc.model_copy(update={
                "visual_type": VisualType.sketchbook_card,
                "scene_kind": sc.scene_kind or "paragraph_card",
            })
            is_sketchbook_scene = True

        # Resume: hash-based skip. Compute the expected hash from the
        # scene's current inputs and compare against the cached card's
        # sidecar manifest. Anything that affects rendered pixels (visual
        # type, prompt, on_screen_text, renderer colours / font, source
        # crop SHA) is in the hash, so cache invalidation is precise.
        if resume and out_path.exists():
            src_path = _source_path_for(sc)
            src_sha = sha256_of(src_path)
            expected = _visual_input_hash(sc, rcfg, icfg,
                                          source_asset_path=src_path,
                                          source_asset_sha=src_sha)
            if manifest_matches(out_path, expected):
                sc = sc.model_copy(update={
                    "visual_asset_paths": [str(out_path)],
                    "status": (SceneStatus.visual_done
                               if sc.status == SceneStatus.audio_done
                               else sc.status),
                })
                new_scenes.append(sc)
                continue
        if sc.status == SceneStatus.failed:
            new_scenes.append(sc)
            continue

        # Sketchbook fast-path. Render via the deterministic Pillow
        # SketchbookRenderer for any scene that has scene_kind set.
        # source_crop scenes also fall through here — their layout
        # payload contains the crop_path produced by a later helper
        # (or it falls back to a paragraph card cleanly).
        if is_sketchbook_scene and sketchbook_renderer is not None:
            try:
                sketchbook_renderer.render(sc, out_path)
            except Exception as e:
                # Fallback to plain card renderer rather than fail the
                # pipeline — sketchbook visuals are best-effort.
                try:
                    renderer.render_scene(sc, out_path)
                    sc = sc.model_copy(update={
                        "last_error": f"sketchbook render fallback: {e!r}",
                    })
                except Exception as e2:
                    sc = sc.model_copy(update={
                        "status": SceneStatus.failed,
                        "last_error": f"render failed: {e2!r}",
                    })
                    failures.append(sc.scene_id)
                    db.upsert_scene(sc.scene_id, sc.chapter_id,
                                    sc.status.value, sc.input_hash,
                                    sc.model_dump(mode="json"),
                                    last_error=sc.last_error, bump_retry=True)
                    new_scenes.append(sc)
                    continue
            consumed_src = _source_path_for(sc)
            consumed_sha = sha256_of(consumed_src)
            manifest_hash = _visual_input_hash(
                sc, rcfg, icfg,
                source_asset_path=consumed_src,
                source_asset_sha=consumed_sha,
            )
            write_manifest(
                out_path,
                stage="visuals",
                scene_id=sc.scene_id,
                input_hash=manifest_hash,
                inputs=_visual_inputs(sc, rcfg, icfg,
                                       source_asset_path=consumed_src,
                                       source_asset_sha=consumed_sha),
                extra={"rendered_visual_type": "sketchbook_card",
                       "scene_kind": sc.scene_kind},
            )
            new_status = (SceneStatus.visual_done
                          if sc.status != SceneStatus.failed else sc.status)
            sc = sc.model_copy(update={
                "visual_asset_paths": [str(out_path)],
                "status": new_status,
            })
            db.register_artifact(out_path, stage="visuals",
                                 scene_id=sc.scene_id, media_type="image/png",
                                 provenance={"input_hash": manifest_hash,
                                             "scene_kind": sc.scene_kind})
            db.upsert_scene(sc.scene_id, sc.chapter_id, sc.status.value,
                            sc.input_hash, sc.model_dump(mode="json"))
            new_scenes.append(sc)
            continue

        # 1) For generated_image scenes, run SDXL into assets/generated/<id>.png.
        #    The SDXL source has its own manifest (independent of the final
        #    card manifest) — without it, changing image.model / steps /
        #    guidance after a first run would re-render the card but reuse
        #    the stale PNG underneath. The final card manifest would then
        #    record the new image config alongside the old image's SHA, a
        #    "false valid" cache hit. With this manifest the cached source
        #    invalidates as soon as any SDXL-pixel-affecting field changes.
        if sc.visual_type == VisualType.generated_image:
            gen_path = p["generated_dir"] / f"{sc.scene_id}.png"
            source_inputs: dict[str, Any] = {
                "schema": "generated_source_v1",
                "scene_id": sc.scene_id,
                "visual_prompt": sc.visual_prompt,
                "image": _image_config_signature(icfg),
                "resolution": [int(res[0]), int(res[1])],
            }
            source_hash = hash_inputs("generated_source_v1", source_inputs)
            needs_generate = not manifest_matches(gen_path, source_hash)
            if not needs_generate:
                sc = sc.model_copy(update={"visual_source_paths": [str(gen_path)]})
            else:
                if image_provider is None and not image_provider_failed:
                    try:
                        image_provider = make_image_provider(icfg)
                    except Exception:
                        image_provider_failed = True
                sdxl_src = (
                    _generate_sdxl_asset(sc, gen_path, image_provider,
                                         (int(res[0]), int(res[1])))
                    if image_provider is not None
                    else None
                )
                _drop_gpu_scratch()
                if sdxl_src:
                    write_manifest(
                        Path(sdxl_src),
                        stage="visual_source",
                        scene_id=sc.scene_id,
                        input_hash=source_hash,
                        inputs=source_inputs,
                        extra={"image_model": icfg.get("model")},
                    )
                    sc = sc.model_copy(update={"visual_source_paths": [sdxl_src]})
                    db.register_artifact(Path(sdxl_src), stage="visuals",
                                         scene_id=sc.scene_id,
                                         media_type="image/png",
                                         provenance={"role": "sdxl_source",
                                                     "model": icfg.get("model"),
                                                     "source_hash": source_hash})
                else:
                    # Downgrade so card renderer takes over.
                    sc = sc.model_copy(update={"visual_type": VisualType.bullet_card})

        # 2) PDF-image visuals require the source crop to exist on disk.
        if sc.visual_type == VisualType.pdf_image:
            src = _source_path_for(sc)
            if not src or not Path(src).exists():
                sc = sc.model_copy(update={"visual_type": VisualType.bullet_card})

        # 3) Render the final card via CardRenderer.
        try:
            renderer.render_scene(sc, out_path)
        except Exception as e:
            try:
                fb = sc.model_copy(update={"visual_type": VisualType.bullet_card,
                                           "visual_source_paths": []})
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

        # Manifest captures the inputs that actually mattered for *this*
        # render — including the SHA of the upstream image so a freshly
        # regenerated SDXL output or swapped PDF crop invalidates next
        # time. Hash uses the same _source_path_for() helper as the
        # resume short-circuit above, so first-render and skip-on-resume
        # compute identical fingerprints.
        consumed_src = _source_path_for(sc)
        consumed_sha = sha256_of(consumed_src)
        manifest_hash = _visual_input_hash(sc, rcfg, icfg,
                                           source_asset_path=consumed_src,
                                           source_asset_sha=consumed_sha)
        write_manifest(
            out_path,
            stage="visuals",
            scene_id=sc.scene_id,
            input_hash=manifest_hash,
            inputs=_visual_inputs(sc, rcfg, icfg,
                                  source_asset_path=consumed_src,
                                  source_asset_sha=consumed_sha),
            extra={"rendered_visual_type": sc.visual_type.value},
        )

        # visual_asset_paths[0] is the final renderable card; the source
        # input stays in visual_source_paths so a future match_visuals
        # or resume can still see it.
        new_status = (SceneStatus.visual_done
                      if sc.status != SceneStatus.failed else sc.status)
        sc = sc.model_copy(update={
            "visual_asset_paths": [str(out_path)],
            "status": new_status,
        })
        db.register_artifact(out_path, stage="visuals", scene_id=sc.scene_id,
                             media_type="image/png",
                             provenance={"input_hash": manifest_hash})
        db.upsert_scene(sc.scene_id, sc.chapter_id, sc.status.value,
                        sc.input_hash, sc.model_dump(mode="json"))
        new_scenes.append(sc)

    graph = graph.model_copy(update={"scenes": new_scenes})
    atomic_write_json(p["scene_graph"], graph.model_dump(mode="json"))
    db.finish_stage("visuals", [str(p["scene_graph"])])
    if failures:
        db.log_error("visuals", f"visual render failed for scenes: {failures}")
    return graph
