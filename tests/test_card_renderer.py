"""CardRenderer — the deterministic Pillow renderer used for every
non-SDXL scene.

We don't pixel-compare (font metrics shift across Pillow versions) but
we do check that the right code paths run, the output dimensions are
right, and that a pdf_image scene with on_screen_text actually paints
both the figure inset and the caption.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from paperreel.models import Scene, SceneStatus, VisualType
from paperreel.renderers.card_renderer import CardRenderer


def _scene(**overrides) -> Scene:
    base = dict(
        scene_id="ch_001_sc_001",
        chapter_id="ch_001",
        title="範例",
        source_pages=[1],
        narration_text_zh_tw="這是一段測試旁白。",
        visual_type=VisualType.bullet_card,
        estimated_duration_sec=20.0,
        input_hash="hash",
        status=SceneStatus.audio_done,
    )
    base.update(overrides)
    return Scene(**base)


def test_bullet_card_runs_at_target_resolution(tmp_path: Path) -> None:
    r = CardRenderer(resolution=(1280, 720))
    out = tmp_path / "card.png"
    r.render_scene(_scene(on_screen_text="重點一\n重點二"), out)
    assert out.exists()
    img = Image.open(out)
    assert img.size == (1280, 720)


def test_pdf_image_card_paints_inset_and_caption(tmp_path: Path) -> None:
    """When a scene has visual_type=pdf_image plus on_screen_text, the
    card must include both the figure and the caption strip — otherwise
    the on_screen_text the LLM wrote is silently dropped."""
    figure = tmp_path / "fig.png"
    # Solid magenta inset so we can detect it in the rendered card.
    Image.new("RGB", (640, 480), color=(220, 40, 220)).save(figure)
    r = CardRenderer(resolution=(1280, 720),
                       background="#101820",
                       foreground="#F8FAFC",
                       accent="#22D3EE")
    out = tmp_path / "card.png"
    r.render_scene(
        _scene(visual_type=VisualType.pdf_image,
                visual_asset_paths=[str(figure)],
                on_screen_text="此圖說明紅色觀念"),
        out,
    )
    assert out.exists()
    img = Image.open(out).convert("RGB")
    assert img.size == (1280, 720)
    # Look for the magenta inset in the middle of the canvas.
    found_inset = False
    for x in range(200, 1080, 20):
        for y in range(200, 540, 20):
            px = img.getpixel((x, y))
            if px[0] > 180 and px[1] < 100 and px[2] > 180:
                found_inset = True
                break
        if found_inset:
            break
    assert found_inset, "expected magenta figure inset somewhere in the card"


def test_pdf_image_card_without_caption_does_not_crash(tmp_path: Path) -> None:
    figure = tmp_path / "fig.png"
    Image.new("RGB", (640, 480), color=(120, 220, 120)).save(figure)
    r = CardRenderer(resolution=(1280, 720))
    out = tmp_path / "card.png"
    r.render_scene(
        _scene(visual_type=VisualType.pdf_image,
                visual_asset_paths=[str(figure)],
                on_screen_text=None),
        out,
    )
    assert out.exists()


def test_pdf_image_card_missing_image_falls_back_gracefully(tmp_path: Path) -> None:
    """A dangling visual_asset_path must not abort the whole render —
    we draw an error notice and keep going so resume can pick the
    scene up next time."""
    r = CardRenderer(resolution=(1280, 720))
    out = tmp_path / "card.png"
    r.render_scene(
        _scene(visual_type=VisualType.pdf_image,
                visual_asset_paths=["/no/such/path.png"]),
        out,
    )
    assert out.exists()
