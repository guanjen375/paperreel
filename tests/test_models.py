from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from paperreel.models import (Scene, SceneGraph, SceneStatus, ScriptScene,
                               VisualType)


def _make_scene(**overrides) -> Scene:
    defaults = dict(
        scene_id="ch_001_sc_001",
        chapter_id="ch_001",
        title="範例 scene",
        source_pages=[1, 2],
        narration_text_zh_tw="這是一段測試用旁白。",
        visual_type=VisualType.bullet_card,
        estimated_duration_sec=30.0,
        input_hash="deadbeef",
    )
    defaults.update(overrides)
    return Scene(**defaults)


def test_scene_round_trip_json() -> None:
    s = _make_scene()
    blob = s.model_dump_json()
    loaded = Scene.model_validate(json.loads(blob))
    assert loaded.scene_id == s.scene_id
    assert loaded.visual_type == VisualType.bullet_card
    assert loaded.status == SceneStatus.pending


def test_scene_status_transitions_via_model_copy() -> None:
    s = _make_scene()
    s2 = s.model_copy(update={"status": SceneStatus.audio_done,
                              "audio_path": "/x.wav",
                              "actual_duration_sec": 27.4})
    assert s2.status == SceneStatus.audio_done
    assert s2.audio_path == "/x.wav"
    assert s2.actual_duration_sec == 27.4


def test_scene_requires_input_hash() -> None:
    with pytest.raises(ValidationError):
        Scene(  # type: ignore[call-arg]
            scene_id="x", chapter_id="c",
            title="t", source_pages=[1],
            narration_text_zh_tw="x",
            visual_type=VisualType.bullet_card,
            estimated_duration_sec=10.0,
            # missing input_hash
        )


def test_scene_graph_groups_scenes() -> None:
    g = SceneGraph(project="demo", target_minutes=12.0,
                   scenes=[_make_scene(), _make_scene(scene_id="ch_001_sc_002")])
    assert len(g.scenes) == 2
    assert g.target_minutes == 12.0


def test_script_scene_accepts_string_visual_type() -> None:
    ss = ScriptScene(scene_id="x", chapter_id="c", title="t",
                     source_pages=[1], narration_text_zh_tw="x",
                     visual_type="recap",
                     estimated_duration_sec=30.0)
    assert ss.visual_type == VisualType.recap

def test_old_scene_artifact_without_visual_fields_loads() -> None:
    old_blob = {
        "scene_id": "old",
        "chapter_id": "ch_001",
        "title": "舊場景",
        "source_pages": [1],
        "narration_text_zh_tw": "舊旁白",
        "visual_type": "sketchbook_card",
        "estimated_duration_sec": 20.0,
        "input_hash": "oldhash",
        "scene_kind": "paragraph_card",
    }
    loaded = Scene.model_validate(old_blob)
    assert loaded.visual_anchor is None
    assert loaded.screen_plan is None
    assert loaded.visual_source_paths == []


def test_visual_first_scene_fields_round_trip() -> None:
    s = _make_scene(
        visual_type=VisualType.sketchbook_card,
        scene_kind="source_visual_explainer",
        visual_anchor={
            "page": 2,
            "image_path": "/tmp/source.png",
            "visual_role": "source_photo",
            "why_this_visual": "example",
        },
        screen_plan={
            "headline": "看來源圖",
            "callouts": ["看主體", "找差異"],
            "layout_hint": "source_visual_explainer",
        },
    )
    loaded = Scene.model_validate(json.loads(s.model_dump_json()))
    assert loaded.visual_anchor is not None
    assert loaded.visual_anchor.page == 2
    assert loaded.screen_plan is not None
    assert loaded.screen_plan.callouts == ["看主體", "找差異"]

