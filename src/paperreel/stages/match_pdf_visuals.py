"""Stage 4b — assign extracted PDF figures to scenes.

Runs between ``scenes`` and ``audio``. For each scene whose source
pages contain candidate images from the ingest stage, we score the
images by:

1. *Page proximity* — figures on a scene's primary source page beat
   figures on adjacent pages.
2. *Caption overlap* — caption hint sharing tokens with the scene
   title / on_screen_text / visual_hint scores higher.
3. *Pixel area* — bigger figures (usually proper plots / diagrams)
   beat tiny decorative icons that already cleared the
   ``min_image_pixels`` filter.
4. *Reuse penalty* — once a figure is assigned to one scene, it gets
   a small score penalty so other scenes prefer fresh figures.

The best-scoring figure (if it clears a minimum threshold) is attached
to the scene's ``visual_asset_paths`` and the scene's ``visual_type``
is upgraded to :attr:`VisualType.pdf_image` so the renderer treats it
as a figure-card. Scenes with no good match are left untouched.

Disabled when the project config has ``visuals.prefer_pdf_figures:
false``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..hashing import hash_inputs
from ..io_utils import atomic_write_json, read_json
from ..models import (ChunkedSources, PdfImage, Scene, SceneGraph,
                       VisualType)
from ..state import StateDB


_TOKEN_RE = re.compile(r"[\w一-鿿㐀-䶿]+", re.UNICODE)


def paths_for(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    return {
        "scene_graph": root / "intermediate" / "scene_graph.json",
        "chunked": root / "intermediate" / "chunked_sources.json",
        "matches": root / "intermediate" / "pdf_visual_matches.json",
    }


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2}


def _score_figure(
    figure: PdfImage,
    scene: Scene,
    *,
    primary_page: int,
    times_used: int,
) -> float:
    """Return a unitless score; higher is better. Negative scores get
    discarded by :func:`_pick_best`."""
    # Page proximity — primary page gets a big boost, then ±1, then ±2.
    page_dist = abs(figure.page - primary_page)
    if page_dist == 0:
        page_score = 5.0
    elif page_dist == 1:
        page_score = 2.5
    elif page_dist == 2:
        page_score = 1.0
    else:
        page_score = 0.0

    # Caption token overlap with the scene's text fields.
    scene_tokens = (_tokens(scene.title)
                    | _tokens(scene.on_screen_text)
                    | _tokens(scene.visual_prompt))
    caption_tokens = _tokens(figure.caption_hint)
    if scene_tokens and caption_tokens:
        overlap = len(scene_tokens & caption_tokens)
        caption_score = min(3.0, overlap * 0.6)
    else:
        caption_score = 0.0

    # Pixel area — log-ish bonus so a 4 Mpx plot beats a 40 kpx icon
    # but doesn't drown out caption matches.
    pixel_score = min(2.0, max(0.0, (figure.pixel_count - 40_000) / 600_000.0))

    # Reuse penalty so we spread figures across scenes instead of every
    # scene latching onto the same hero image.
    reuse_penalty = 0.7 * times_used

    return page_score + caption_score + pixel_score - reuse_penalty


def _pick_best(
    scene: Scene, figures_by_page: dict[int, list[PdfImage]],
    used_counts: dict[str, int],
    *,
    min_score: float,
) -> tuple[PdfImage, float] | None:
    """Return ``(figure, score)`` for the best candidate, or None if no
    candidate clears ``min_score``."""
    primary = scene.source_pages[0] if scene.source_pages else None
    if primary is None:
        return None
    candidates: list[PdfImage] = []
    for page in scene.source_pages:
        candidates.extend(figures_by_page.get(page, []))
        # Allow neighbouring pages so a scene that spans p.12-13 can
        # still pick up a figure that landed on p.14.
        candidates.extend(figures_by_page.get(page + 1, []))
        candidates.extend(figures_by_page.get(page - 1, []))
    if not candidates:
        return None
    # Deduplicate (a figure on a shared page would otherwise appear twice).
    seen: dict[str, PdfImage] = {}
    for c in candidates:
        seen.setdefault(c.image_id, c)
    best: tuple[PdfImage, float] | None = None
    for fig in seen.values():
        score = _score_figure(
            fig, scene,
            primary_page=primary,
            times_used=used_counts.get(fig.image_id, 0),
        )
        if score < min_score:
            continue
        if best is None or score > best[1]:
            best = (fig, score)
    return best


def _figures_by_page(images: list[PdfImage]) -> dict[int, list[PdfImage]]:
    out: dict[int, list[PdfImage]] = {}
    for im in images:
        out.setdefault(im.page, []).append(im)
    return out


def run(*, project_root: str | Path, db: StateDB,
        config: dict) -> SceneGraph:
    p = paths_for(project_root)
    graph = SceneGraph.model_validate(read_json(p["scene_graph"]))
    sources = ChunkedSources.model_validate(read_json(p["chunked"]))

    vis_cfg = config.get("visuals", {})
    prefer_figures = bool(vis_cfg.get("prefer_pdf_figures", True))
    min_score = float(vis_cfg.get("pdf_figure_min_score", 3.0))

    input_hash = hash_inputs(
        "match_visuals_v1",
        sources.pdf_sha256,
        [s.scene_id for s in graph.scenes],
        [(im.image_id, im.sha256) for im in sources.images],
        prefer_figures, min_score,
    )
    db.start_stage("match_visuals", input_hash)

    matches_log: list[dict[str, Any]] = []
    if not prefer_figures or not sources.images:
        atomic_write_json(p["matches"], {
            "schema": "match_visuals_v1",
            "skipped": True,
            "reason": ("prefer_pdf_figures=false" if not prefer_figures
                       else "no images in PDF"),
            "matches": [],
        })
        db.finish_stage("match_visuals", [str(p["matches"])])
        return graph

    figures_by_page = _figures_by_page(sources.images)
    used_counts: dict[str, int] = {}
    new_scenes: list[Scene] = []
    for sc in graph.scenes:
        # Skip scenes that already point at a real (external) asset —
        # we shouldn't overwrite something the LLM / a previous stage
        # explicitly chose.
        already_assigned = bool(sc.visual_asset_paths)
        if already_assigned:
            new_scenes.append(sc)
            continue
        if sc.visual_type in (VisualType.title_card, VisualType.recap,
                               VisualType.quiz):
            # These templates carry their own meaning; we don't want to
            # swap a recap card out for a random figure.
            new_scenes.append(sc)
            continue
        best = _pick_best(sc, figures_by_page, used_counts,
                           min_score=min_score)
        if best is None:
            new_scenes.append(sc)
            continue
        figure, score = best
        used_counts[figure.image_id] = used_counts.get(figure.image_id, 0) + 1
        sc = sc.model_copy(update={
            "visual_type": VisualType.pdf_image,
            "visual_asset_paths": [figure.path],
        })
        matches_log.append({
            "scene_id": sc.scene_id,
            "image_id": figure.image_id,
            "page": figure.page,
            "score": round(score, 3),
            "caption_hint": figure.caption_hint,
            "image_path": figure.path,
        })
        db.upsert_scene(sc.scene_id, sc.chapter_id, sc.status.value,
                        sc.input_hash, sc.model_dump(mode="json"))
        new_scenes.append(sc)

    graph = graph.model_copy(update={"scenes": new_scenes})
    atomic_write_json(p["scene_graph"], graph.model_dump(mode="json"))
    atomic_write_json(p["matches"], {
        "schema": "match_visuals_v1",
        "skipped": False,
        "matches": matches_log,
    })
    db.register_artifact(p["matches"], stage="match_visuals",
                         media_type="application/json")
    db.finish_stage("match_visuals", [str(p["matches"])])
    return graph
