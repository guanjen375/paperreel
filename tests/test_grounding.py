"""Grounding validator — every factual sketchbook scene must trace
back to the ingested PDF text."""
from __future__ import annotations

import pytest

from paperreel.models import EvidenceSpan, Fact, ScriptScene, VisualType
from paperreel.utils import grounding


def _scene(scene_kind: str, **kwargs) -> ScriptScene:
    base = dict(
        scene_id="ch_001_sc_001",
        chapter_id="ch_001",
        title="測試 scene",
        source_pages=[3],
        narration_text_zh_tw="測試 narration",
        on_screen_text=None,
        visual_hint=None,
        visual_type=VisualType.sketchbook_card,
        estimated_duration_sec=22.0,
        scene_kind=scene_kind,
        facts=[],
        evidence_spans=[],
        layout_payload={},
        importance="high",
    )
    base.update(kwargs)
    return ScriptScene(**base)


PAGE_TEXT = {
    3: "本合約約定，雙方應於 45 天內完成付款，違約金為合約金額的 30%。",
    4: "若一方違約，他方有權終止合約並請求賠償。",
}


def test_factual_scene_without_evidence_fails() -> None:
    sc = _scene("deadline_timeline")
    issues = grounding.validate_scene(
        sc, page_text=PAGE_TEXT,
        min_quote_ratio=0.55,
        require_evidence_for_facts=True,
    )
    assert any(i.code == "missing_evidence" for i in issues)


def test_factual_scene_with_matching_quote_passes() -> None:
    sc = _scene(
        "deadline_timeline",
        evidence_spans=[EvidenceSpan(
            page=3, quote="雙方應於 45 天內完成付款",
            label="期限", value="45 天",
        )],
        facts=[Fact(label="期限", value="45 天",
                     importance="high", evidence_index=0)],
    )
    issues = grounding.validate_scene(
        sc, page_text=PAGE_TEXT,
        min_quote_ratio=0.55,
        require_evidence_for_facts=True,
    )
    assert issues == []


def test_quote_not_in_page_fails() -> None:
    sc = _scene(
        "penalty_table",
        evidence_spans=[EvidenceSpan(
            page=3, quote="這段文字根本不在任何頁面裡面",
            label="罰則", value="無稽之談",
        )],
    )
    issues = grounding.validate_scene(
        sc, page_text=PAGE_TEXT,
        min_quote_ratio=0.55,
        require_evidence_for_facts=True,
    )
    assert any(i.code == "quote_mismatch" for i in issues)


def test_bad_page_number_flagged() -> None:
    sc = _scene(
        "checklist",
        evidence_spans=[EvidenceSpan(
            page=99, quote="本合約", label="應辦",
        )],
    )
    issues = grounding.validate_scene(
        sc, page_text=PAGE_TEXT,
        min_quote_ratio=0.55,
        require_evidence_for_facts=True,
    )
    assert any(i.code == "bad_page" for i in issues)


def test_quote_match_ratio_handles_punctuation_drift() -> None:
    """LLM polish often swaps full-width punctuation; ratio should
    still come back >= 0.9 for substantively matching text."""
    quote = "雙方,應於 45天 內 完成付款"
    page = "本合約約定，雙方應於 45 天內完成付款，違約金為 30%。"
    ratio = grounding.quote_match_ratio(quote, page)
    assert ratio >= 0.6


def test_cover_scene_does_not_require_evidence() -> None:
    sc = _scene("cover", source_pages=[1])
    issues = grounding.validate_scene(
        sc, page_text=PAGE_TEXT,
        min_quote_ratio=0.55,
        require_evidence_for_facts=True,
    )
    assert issues == []


def test_validate_scenes_collects_all_issues() -> None:
    scenes = [
        _scene("deadline_timeline"),
        _scene(
            "deadline_timeline",
            scene_id="ch_001_sc_002",
            evidence_spans=[EvidenceSpan(
                page=3, quote="雙方應於 45 天內完成付款",
                label="期限", value="45 天",
            )],
        ),
    ]
    issues = grounding.validate_scenes(
        scenes, page_text=PAGE_TEXT,
        min_quote_ratio=0.55,
        require_evidence_for_facts=True,
    )
    # First scene fails (no evidence), second passes.
    by_scene = {i.scene_id for i in issues}
    assert "ch_001_sc_001" in by_scene
    assert "ch_001_sc_002" not in by_scene
