"""Stage 4b: assign extracted PDF figures to scenes.

The match stage exists because the LLM script writer can't see the
actual figures in the PDF — it would rarely choose ``visual_type=
pdf_image`` even when a relevant figure exists. This stage joins
``scene.source_pages`` against ``ChunkedSources.images`` and upgrades
the scene's visual_type when a good candidate exists.

Tests pin the contract:
- a scene on the figure's page gets the figure
- scenes that already have an asset are left alone
- title / recap / quiz cards are never overwritten
- the same figure isn't blindly used by every scene
- disabling via config short-circuits without touching anything
"""
from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image

from paperreel.io_utils import atomic_write_json, read_json
from paperreel.models import (Scene, SceneGraph, SceneStatus, VisualType)
from paperreel.stages import match_pdf_visuals
from paperreel.state import StateDB


def _scene(scene_id: str, *, chapter: str = "ch_001",
           source_pages: list[int] | None = None,
           visual_type: VisualType = VisualType.bullet_card,
           visual_source_paths: list[str] | None = None,
           visual_asset_paths: list[str] | None = None,
           title: str = "Test scene",
           on_screen_text: str | None = None,
           visual_prompt: str = "") -> Scene:
    return Scene(
        scene_id=scene_id, chapter_id=chapter, title=title,
        source_pages=source_pages or [1],
        narration_text_zh_tw="這是測試旁白。" * 5,
        visual_prompt=visual_prompt,
        visual_type=visual_type,
        on_screen_text=on_screen_text,
        visual_source_paths=visual_source_paths or [],
        visual_asset_paths=visual_asset_paths or [],
        estimated_duration_sec=25.0,
        status=SceneStatus.pending,
        input_hash="hash",
    )


def _setup_project(project_dir: Path, *,
                    scenes: list[Scene],
                    images_pdf_fixture,
                    tmp_path: Path,
                    test_cfg: dict) -> StateDB:
    """Run ingest on a fixture PDF to populate chunked_sources.images,
    then write a synthetic scene_graph.json with the given scenes so
    we can exercise the match stage in isolation."""
    from paperreel.stages import ingest_pdf
    db = StateDB(project_dir / "state.sqlite")
    ingest_pdf.run(pdf_path=images_pdf_fixture, project_root=project_dir,
                    db=db, config=test_cfg)
    graph = SceneGraph(project="t", target_minutes=5.0, scenes=scenes)
    atomic_write_json(
        project_dir / "intermediate" / "scene_graph.json",
        graph.model_dump(mode="json"),
    )
    return db


def _two_figure_pdf(tmp_path: Path) -> Path:
    """A 4-page PDF with one figure on p.2 and one on p.4 — enough to
    test page-based matching, reuse penalty, and "no figure on this
    scene's pages" behaviour."""
    pdf_path = tmp_path / "two_figures.pdf"
    doc = fitz.open()
    # Two distinct images so we can tell which figure got picked.
    fig_a = tmp_path / "fig_a.png"
    Image.new("RGB", (400, 300), color=(40, 180, 40)).save(fig_a)
    fig_b = tmp_path / "fig_b.png"
    Image.new("RGB", (500, 400), color=(40, 40, 220)).save(fig_b)

    for pno in range(1, 5):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 80), f"Page {pno} body text.",
                          fontsize=12, fontname="helv")
        if pno == 2:
            page.insert_image(fitz.Rect(100, 100, 400, 325), filename=str(fig_a))
            page.insert_text((100, 360), "Figure A: green diagram.",
                              fontsize=11, fontname="helv")
        if pno == 4:
            page.insert_image(fitz.Rect(100, 100, 400, 325), filename=str(fig_b))
            page.insert_text((100, 360), "Figure B: blue chart.",
                              fontsize=11, fontname="helv")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


# --- core matching ---------------------------------------------------------

def test_match_assigns_figure_on_source_page(
    project_dir: Path, tmp_path: Path, test_cfg: dict
) -> None:
    pdf = _two_figure_pdf(tmp_path)
    sc_on_fig_page = _scene("ch_001_sc_001", source_pages=[2],
                              title="Diagram",
                              visual_prompt="green diagram")
    db = _setup_project(project_dir, scenes=[sc_on_fig_page],
                         images_pdf_fixture=pdf,
                         tmp_path=tmp_path, test_cfg=test_cfg)
    g = match_pdf_visuals.run(project_root=project_dir, db=db, config=test_cfg)
    db.close()
    out = g.scenes[0]
    assert out.visual_type == VisualType.pdf_image
    # The matched PDF figure goes into visual_source_paths (the
    # renderer's input). visual_asset_paths (the renderer's output)
    # stays empty until render_visuals runs.
    assert len(out.visual_source_paths) == 1
    assert Path(out.visual_source_paths[0]).exists()
    assert out.visual_asset_paths == []
    # Should have picked the green figure (page 2), not the blue one (page 4).
    img = Image.open(out.visual_source_paths[0]).convert("RGB")
    px = img.getpixel((img.width // 2, img.height // 2))
    assert px[1] > px[0] and px[1] > px[2], f"expected green hero pixel, got {px}"


def test_match_leaves_scene_alone_when_no_figure_on_pages(
    project_dir: Path, tmp_path: Path, test_cfg: dict
) -> None:
    pdf = _two_figure_pdf(tmp_path)
    # Scene on page 1: figures are on p.2 and p.4 (adjacent / 3 away).
    # Carefully avoid words that appear in either caption ("Figure",
    # "green", "diagram", "blue", "chart") so caption overlap can't
    # tip an adjacent figure over the score threshold.
    sc = _scene("ch_001_sc_001", source_pages=[1],
                  title="Introduction overview")
    db = _setup_project(project_dir, scenes=[sc],
                         images_pdf_fixture=pdf,
                         tmp_path=tmp_path, test_cfg=test_cfg)
    g = match_pdf_visuals.run(project_root=project_dir, db=db, config=test_cfg)
    db.close()
    # Page 1 is adjacent to page 2 (which has fig_a). Adjacency boost is
    # 2.5; with no caption overlap and small pixel bonus, total stays
    # under the default min_score=3.0, so the scene must be untouched.
    out = g.scenes[0]
    assert out.visual_type == VisualType.bullet_card
    assert out.visual_source_paths == []


def test_match_does_not_overwrite_existing_source(
    project_dir: Path, tmp_path: Path, test_cfg: dict
) -> None:
    """If a scene already has visual_source_paths set (e.g. by a prior
    match or by hand), the stage must not clobber that choice."""
    pdf = _two_figure_pdf(tmp_path)
    sc = _scene("ch_001_sc_001", source_pages=[2],
                  visual_source_paths=["/tmp/some_existing_thing.png"],
                  title="Has source already")
    db = _setup_project(project_dir, scenes=[sc],
                         images_pdf_fixture=pdf,
                         tmp_path=tmp_path, test_cfg=test_cfg)
    g = match_pdf_visuals.run(project_root=project_dir, db=db, config=test_cfg)
    db.close()
    out = g.scenes[0]
    assert out.visual_source_paths == ["/tmp/some_existing_thing.png"]


def test_match_runs_when_only_visual_asset_paths_set(
    project_dir: Path, tmp_path: Path, test_cfg: dict
) -> None:
    """Regression for the self-nesting bug: a scene whose
    visual_asset_paths is populated (from a previous render pass) but
    whose visual_source_paths is empty MUST still get re-matched.
    Otherwise resume loses the upstream figure forever."""
    pdf = _two_figure_pdf(tmp_path)
    sc = _scene("ch_001_sc_001", source_pages=[2],
                  visual_asset_paths=["/some/previously/rendered/card.png"],
                  visual_type=VisualType.pdf_image,
                  title="Diagram")
    db = _setup_project(project_dir, scenes=[sc],
                         images_pdf_fixture=pdf,
                         tmp_path=tmp_path, test_cfg=test_cfg)
    g = match_pdf_visuals.run(project_root=project_dir, db=db, config=test_cfg)
    db.close()
    out = g.scenes[0]
    assert out.visual_source_paths, "must re-match because source was empty"
    assert "pdf_images" in out.visual_source_paths[0]


def test_match_never_overwrites_title_or_recap(
    project_dir: Path, tmp_path: Path, test_cfg: dict
) -> None:
    """Title cards, recap cards, and quiz cards carry their own meaning
    — we can't replace them with an unrelated figure even if one's on
    the same page."""
    pdf = _two_figure_pdf(tmp_path)
    title_sc = _scene("ch_001_sc_001", source_pages=[2],
                        visual_type=VisualType.title_card, title="Opening")
    recap_sc = _scene("ch_001_sc_002", source_pages=[2],
                        visual_type=VisualType.recap, title="Recap")
    quiz_sc = _scene("ch_001_sc_003", source_pages=[2],
                       visual_type=VisualType.quiz, title="Quiz")
    db = _setup_project(project_dir, scenes=[title_sc, recap_sc, quiz_sc],
                         images_pdf_fixture=pdf,
                         tmp_path=tmp_path, test_cfg=test_cfg)
    g = match_pdf_visuals.run(project_root=project_dir, db=db, config=test_cfg)
    db.close()
    types = {s.scene_id: s.visual_type for s in g.scenes}
    assert types["ch_001_sc_001"] == VisualType.title_card
    assert types["ch_001_sc_002"] == VisualType.recap
    assert types["ch_001_sc_003"] == VisualType.quiz


def test_match_assigns_distinct_figures_across_scenes(
    project_dir: Path, tmp_path: Path, test_cfg: dict
) -> None:
    """Each scene sits on its own figure's page — they MUST each land
    on the right figure (sanity check that we're not just blindly
    handing out the first one)."""
    pdf = _two_figure_pdf(tmp_path)
    sc1 = _scene("ch_001_sc_001", source_pages=[2], title="Diagram A")
    sc2 = _scene("ch_001_sc_002", source_pages=[4], title="Chart B")
    db = _setup_project(project_dir, scenes=[sc1, sc2],
                         images_pdf_fixture=pdf,
                         tmp_path=tmp_path, test_cfg=test_cfg)
    g = match_pdf_visuals.run(project_root=project_dir, db=db, config=test_cfg)
    db.close()
    by_id = {s.scene_id: s for s in g.scenes}
    p1 = by_id["ch_001_sc_001"].visual_source_paths[0]
    p2 = by_id["ch_001_sc_002"].visual_source_paths[0]
    assert p1 != p2
    # Distinguishable by colour: fig_a is green, fig_b is blue.
    img1 = Image.open(p1).convert("RGB")
    img2 = Image.open(p2).convert("RGB")
    px1 = img1.getpixel((img1.width // 2, img1.height // 2))
    px2 = img2.getpixel((img2.width // 2, img2.height // 2))
    assert px1[1] > px1[2], f"sc1 expected green, got {px1}"
    assert px2[2] > px2[1], f"sc2 expected blue, got {px2}"


def test_match_score_includes_reuse_penalty() -> None:
    """White-box: the score helper should drop when a figure has
    already been used. This is what prevents one hero figure from
    smothering every other scene's match in a long document."""
    from paperreel.models import PdfImage
    from paperreel.stages.match_pdf_visuals import _score_figure
    fig = PdfImage(
        image_id="img_1", page=2, path="/x.png",
        width=400, height=300, pixel_count=120_000,
        sha256="deadbeef", caption_hint="diagram", bbox=None,
    )
    sc = _scene("ch_001_sc_001", source_pages=[2], title="anything")
    fresh = _score_figure(fig, sc, primary_page=2, times_used=0)
    used = _score_figure(fig, sc, primary_page=2, times_used=1)
    assert used < fresh
    assert (fresh - used) > 0.5  # roughly the 0.7 reuse_penalty constant


# --- config-driven behaviour ----------------------------------------------

def test_prefer_pdf_figures_false_short_circuits(
    project_dir: Path, tmp_path: Path, test_cfg: dict
) -> None:
    pdf = _two_figure_pdf(tmp_path)
    sc = _scene("ch_001_sc_001", source_pages=[2])
    db = _setup_project(project_dir, scenes=[sc],
                         images_pdf_fixture=pdf,
                         tmp_path=tmp_path, test_cfg=test_cfg)
    cfg = {**test_cfg, "visuals": {"prefer_pdf_figures": False}}
    g = match_pdf_visuals.run(project_root=project_dir, db=db, config=cfg)
    db.close()
    out = g.scenes[0]
    assert out.visual_type == VisualType.bullet_card
    assert out.visual_source_paths == []
    # And the run still wrote the matches log so status reporting works.
    log = read_json(project_dir / "intermediate" / "pdf_visual_matches.json")
    assert log["skipped"] is True


def test_resume_does_not_self_nest_pdf_image_card(
    project_dir: Path, tmp_path: Path, test_cfg: dict
) -> None:
    """End-to-end regression: run match_visuals + render_visuals twice.
    On the second pass the matched PDF figure must still be the source
    (not the previously-rendered card), so the renderer doesn't embed
    its own output inside a new card."""
    from paperreel.stages import render_visuals

    pdf = _two_figure_pdf(tmp_path)
    sc = _scene("ch_001_sc_001", source_pages=[2], title="Diagram",
                  visual_prompt="green diagram")
    db = _setup_project(project_dir, scenes=[sc],
                         images_pdf_fixture=pdf,
                         tmp_path=tmp_path, test_cfg=test_cfg)

    # First pass: match + render.
    match_pdf_visuals.run(project_root=project_dir, db=db, config=test_cfg)
    render_visuals.run(project_root=project_dir, db=db, config=test_cfg)

    g_after_first = SceneGraph.model_validate(read_json(
        project_dir / "intermediate" / "scene_graph.json"
    ))
    sc_after_first = g_after_first.scenes[0]
    src_after_first = sc_after_first.visual_source_paths[0]
    asset_after_first = sc_after_first.visual_asset_paths[0]
    assert "pdf_images" in src_after_first
    assert "assets/visuals" in asset_after_first
    assert src_after_first != asset_after_first

    # Second pass: source path must still be the PDF figure, never
    # the rendered card. This is the regression: in the old (single
    # visual_asset_paths field) world, the source would silently become
    # the previously rendered card.
    match_pdf_visuals.run(project_root=project_dir, db=db, config=test_cfg)
    g_after_second = SceneGraph.model_validate(read_json(
        project_dir / "intermediate" / "scene_graph.json"
    ))
    sc_after_second = g_after_second.scenes[0]
    assert sc_after_second.visual_source_paths == [src_after_first]
    assert sc_after_second.visual_type == VisualType.pdf_image

    # And a fresh-config re-render reads from the source, not from the
    # previously rendered card.
    new_cfg = {**test_cfg,
               "renderer": {**test_cfg["renderer"],
                             "background_color": "#FF00FF"}}
    render_visuals.run(project_root=project_dir, db=db, config=new_cfg)
    g_after_rerender = SceneGraph.model_validate(read_json(
        project_dir / "intermediate" / "scene_graph.json"
    ))
    final_src = g_after_rerender.scenes[0].visual_source_paths[0]
    assert final_src == src_after_first, (
        "second render swapped its own output into visual_source_paths — "
        "this is the self-nesting bug"
    )
    db.close()


def test_match_writes_log_for_inspection(
    project_dir: Path, tmp_path: Path, test_cfg: dict
) -> None:
    """The intermediate matches log is what we surface in CLI output and
    in any future quality-repair flow — pin its shape."""
    pdf = _two_figure_pdf(tmp_path)
    sc_match = _scene("ch_001_sc_001", source_pages=[2],
                        title="Diagram explaining green",
                        visual_prompt="green diagram")
    sc_no_match = _scene("ch_001_sc_002", source_pages=[1])
    db = _setup_project(project_dir, scenes=[sc_match, sc_no_match],
                         images_pdf_fixture=pdf,
                         tmp_path=tmp_path, test_cfg=test_cfg)
    match_pdf_visuals.run(project_root=project_dir, db=db, config=test_cfg)
    db.close()
    log = read_json(project_dir / "intermediate" / "pdf_visual_matches.json")
    assert log["skipped"] is False
    matches = log["matches"]
    assert any(m["scene_id"] == "ch_001_sc_001" for m in matches)
    only = next(m for m in matches if m["scene_id"] == "ch_001_sc_001")
    assert only["page"] == 2
    assert only["score"] >= 3.0
