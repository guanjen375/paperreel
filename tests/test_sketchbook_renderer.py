"""SketchbookRenderer — every layout kind must produce a 1280x720
RGB PNG (or whatever resolution we asked for) without raising.

We don't pixel-compare; we just exercise each code path and make sure
the output file exists at the requested dimensions."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from paperreel.models import (EvidenceSpan, Fact, Scene, SceneStatus,
                                VisualType)
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
