"""SketchbookRenderer — every layout kind must produce a 1280x720
RGB PNG (or whatever resolution we asked for) without raising.

We don't pixel-compare; we just exercise each code path and make sure
the output file exists at the requested dimensions."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from paperreel.models import (EvidenceSpan, Fact, Scene, SceneStatus,
                                ScreenPlan, VisualAnchor, VisualType)
from paperreel.renderers.sketchbook_renderer import SketchbookRenderer


def _scene(scene_kind: str, **overrides) -> Scene:
    base = dict(
        scene_id="ch_001_sc_001",
        chapter_id="ch_001",
        title="範例卡片",
        source_pages=[1, 2],
        narration_text_zh_tw="這是測試旁白。",
        visual_type=VisualType.sketchbook_card,
        estimated_duration_sec=22.0,
        input_hash="hash",
        status=SceneStatus.audio_done,
        scene_kind=scene_kind,
    )
    base.update(overrides)
    return Scene(**base)


def _renderer() -> SketchbookRenderer:
    return SketchbookRenderer(resolution=(1280, 720))


def _assert_card(out: Path, expected_size: tuple[int, int]) -> None:
    assert out.exists()
    img = Image.open(out)
    assert img.size == expected_size
    assert img.mode == "RGB"


def test_cover_renders(tmp_path: Path) -> None:
    out = tmp_path / "card.png"
    _renderer().render(
        _scene("cover",
                layout_payload={"eyebrow": "合約導讀",
                                 "subtitle": "共 5 段，4 分鐘",
                                 "doc_kind": "contract"}),
        out,
    )
    _assert_card(out, (1280, 720))


def test_deadline_timeline_renders(tmp_path: Path) -> None:
    out = tmp_path / "card.png"
    _renderer().render(
        _scene("deadline_timeline",
                layout_payload={
                    "events": [
                        {"value": "45 天", "label": "付款期限", "page": 3},
                        {"value": "30 天", "label": "退款期限", "page": 4},
                        {"value": "7 日", "label": "提交文件", "page": 5},
                    ],
                }),
        out,
    )
    _assert_card(out, (1280, 720))


def test_penalty_table_renders(tmp_path: Path) -> None:
    out = tmp_path / "card.png"
    _renderer().render(
        _scene("penalty_table",
                layout_payload={
                    "rows": [
                        {"condition": "30 日前取消", "value": "退 80%", "page": 6},
                        {"condition": "14 日前取消", "value": "退 50%", "page": 6},
                        {"condition": "7 日前取消", "value": "退 20%", "page": 7},
                    ],
                }),
        out,
    )
    _assert_card(out, (1280, 720))


def test_checklist_renders(tmp_path: Path) -> None:
    out = tmp_path / "card.png"
    _renderer().render(
        _scene("checklist",
                layout_payload={
                    "items": [
                        {"text": "繳交護照影本", "page": 8},
                        {"text": "簽署同意書", "page": 9},
                        {"text": "完成保險投保", "page": 10},
                    ],
                }),
        out,
    )
    _assert_card(out, (1280, 720))


def test_risk_warning_renders(tmp_path: Path) -> None:
    out = tmp_path / "card.png"
    _renderer().render(
        _scene("risk_warning",
                layout_payload={
                    "items": [
                        {"text": "若延遲交件將不予退款", "page": 11},
                        {"text": "天候因素風險自負", "page": 12},
                    ],
                }),
        out,
    )
    _assert_card(out, (1280, 720))


def test_do_dont_renders(tmp_path: Path) -> None:
    out = tmp_path / "card.png"
    _renderer().render(
        _scene("do_dont",
                layout_payload={
                    "do": [{"text": "備妥所有證件", "page": 13}],
                    "dont": [{"text": "請勿遲交資料", "page": 13}],
                }),
        out,
    )
    _assert_card(out, (1280, 720))


def test_recap_renders(tmp_path: Path) -> None:
    out = tmp_path / "card.png"
    _renderer().render(
        _scene("recap_card",
                layout_payload={"items": [
                    {"text": "45 天付款", "page": 3},
                    {"text": "罰則三檔", "page": 6},
                    {"text": "保險必辦", "page": 10},
                ]}),
        out,
    )
    _assert_card(out, (1280, 720))


def test_paragraph_card_fallback(tmp_path: Path) -> None:
    out = tmp_path / "card.png"
    _renderer().render(
        _scene("paragraph_card",
                layout_payload={"body": "這是補充說明卡片，"
                                "顯示一段中等長度的旁白。"}),
        out,
    )
    _assert_card(out, (1280, 720))


def test_section_intro_renders(tmp_path: Path) -> None:
    out = tmp_path / "card.png"
    _renderer().render(
        _scene("section_intro",
                title="付款條款",
                layout_payload={"number": "02",
                                 "body": "本段說明付款時間與罰則計算方式。"}),
        out,
    )
    _assert_card(out, (1280, 720))


def test_key_number_renders(tmp_path: Path) -> None:
    out = tmp_path / "card.png"
    _renderer().render(
        _scene("key_number",
                layout_payload={"items": [
                    {"label": "違約金", "value": "30%",
                     "context": "違約金為總額 30%", "page": 6},
                ]}),
        out,
    )
    _assert_card(out, (1280, 720))


def test_unknown_kind_falls_back_to_paragraph(tmp_path: Path) -> None:
    out = tmp_path / "card.png"
    _renderer().render(
        _scene("totally_unknown_kind",
                layout_payload={"body": "fallback narration"}),
        out,
    )
    _assert_card(out, (1280, 720))


def test_source_crop_without_crop_path_uses_quote_card(tmp_path: Path) -> None:
    out = tmp_path / "card.png"
    renderer = _renderer()

    def fail_placeholder(*args, **kwargs):
        raise AssertionError("missing source crop should not render placeholder")

    renderer._placeholder = fail_placeholder  # type: ignore[method-assign]
    renderer.render(
        _scene(
            "source_crop",
            source_pages=[5],
            evidence_spans=[
                EvidenceSpan(page=5, quote="出發前30天內恕無法更動任何名字及艙房分配。"),
            ],
            layout_payload={
                "caption": "沒有裁切圖時，仍應顯示可讀的來源摘錄。",
            },
        ),
        out,
    )
    _assert_card(out, (1280, 720))


def test_dense_timeline_renders_without_horizontal_overlap_layout(tmp_path: Path) -> None:
    out = tmp_path / "dense_timeline.png"
    events = [
        {"value": "出發前89-60 天以上", "label": "取消者收取全額船艙費用之 30% 為取消費用", "page": 1},
        {"value": "出發前59-32 天以上", "label": "取消者收取全額船艙費用之 50% 為取消費用", "page": 1},
        {"value": "出發前31-16 天以上", "label": "取消者收取全額船艙費用之 75% 為取消費用", "page": 1},
        {"value": "出發前44~31 天", "label": "更改名單或艙房分配需付每人新台幣 3,000 元", "page": 1},
        {"value": "出發前45 天", "label": "應繳付全額費用並提供正確名單", "page": 1},
        {"value": "出發前90 天以上", "label": "取消者收取全額訂金為取消費用", "page": 1},
    ]
    _renderer().render(
        _scene("deadline_timeline", layout_payload={"events": events}),
        out,
    )
    _assert_card(out, (1280, 720))


def test_long_checklist_rows_use_dynamic_spacing(tmp_path: Path) -> None:
    out = tmp_path / "long_checklist.png"
    _renderer().render(
        _scene("checklist", layout_payload={"items": [
            {"text": "出發前44~31 天甲方欲更改名單或艙房分配時，需付每人新台幣3,000 元改名手續費", "page": 1},
            {"text": "甲方需於出發前45 天以上提供乙方正確名單及分房，同護照上之正確英文姓名", "page": 1},
            {"text": "甲方需於出發前45 天以上繳交護照資料及行程必要之簽證資料", "page": 1},
        ]}),
        out,
    )
    _assert_card(out, (1280, 720))

def test_source_visual_explainer_renders_large_source_visual(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (900, 520), color=(220, 40, 40)).save(source)
    out = tmp_path / "source_visual.png"
    scene = _scene(
        "source_visual_explainer",
        source_pages=[3],
        visual_source_paths=[str(source)],
        visual_anchor=VisualAnchor(
            page=3, image_path=str(source), visual_role="source_photo",
            caption="光圈範例", nearby_heading="光圈與景深",
            why_this_visual="example photo",
        ),
        screen_plan=ScreenPlan(
            headline="光圈與景深",
            callouts=["看主體", "景深變化", "背景模糊"],
            labels=["景深變化"],
            max_screen_text=60,
            layout_hint="source_visual_explainer",
        ),
        layout_payload={
            "headline": "光圈與景深",
            "image_path": str(source),
            "callouts": [{"text": "看主體"}, {"text": "背景模糊"}],
            "visual_role": "source_photo",
        },
    )
    _renderer().render(scene, out)
    _assert_card(out, (1280, 720))
    img = Image.open(out).convert("RGB")
    red_pixels = sum(
        1 for r, g, b in img.getdata()
        if r > 170 and g < 90 and b < 90
    )
    assert red_pixels > 180_000


def test_comparison_visual_card_renders_two_visuals_with_labels(tmp_path: Path) -> None:
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    Image.new("RGB", (640, 420), color=(40, 180, 60)).save(left)
    Image.new("RGB", (640, 420), color=(40, 80, 220)).save(right)
    out = tmp_path / "comparison.png"
    scene = _scene(
        "comparison_visual_card",
        source_pages=[4, 5],
        visual_anchor=VisualAnchor(
            page=4, image_path=str(left), visual_role="source_photo",
        ),
        screen_plan=ScreenPlan(
            headline="景深比較", labels=["淺景深", "深景深"],
            callouts=["看差異"], layout_hint="comparison_visual_card",
        ),
        layout_payload={
            "headline": "景深比較",
            "visuals": [
                {"image_path": str(left), "label": "淺景深", "page": 4},
                {"image_path": str(right), "label": "深景深", "page": 5},
            ],
        },
    )
    _renderer().render(scene, out)
    _assert_card(out, (1280, 720))
    img = Image.open(out).convert("RGB")
    green_pixels = sum(1 for r, g, b in img.getdata() if g > 140 and r < 90 and b < 110)
    blue_pixels = sum(1 for r, g, b in img.getdata() if b > 160 and r < 90 and g < 130)
    assert green_pixels > 90_000
    assert blue_pixels > 90_000

