"""Duration controller — depth maps to a window; target_minutes overrides
depth; select_scenes trims low-priority scenes to fit the window."""
from __future__ import annotations

from paperreel.models import (EvidenceSpan, Fact, ScriptScene, VisualType)
from paperreel.utils import scene_budget


def _scene(scene_kind: str, *, duration: float = 22.0,
           importance: str = "medium") -> ScriptScene:
    return ScriptScene(
        scene_id=f"ch_001_sc_{scene_kind[:6]}",
        chapter_id="ch_001",
        title="x",
        source_pages=[1],
        narration_text_zh_tw="x",
        visual_type=VisualType.sketchbook_card,
        estimated_duration_sec=duration,
        scene_kind=scene_kind,
        importance=importance,
    )


def test_depth_brief_targets_about_2min() -> None:
    t = scene_budget.resolve_target(depth="brief", target_minutes=None)
    assert 90 <= t.target_seconds <= 150
    assert t.depth == "brief"


def test_depth_standard_targets_about_5min() -> None:
    t = scene_budget.resolve_target(depth="standard", target_minutes=None)
    assert 240 <= t.target_seconds <= 360


def test_depth_deep_targets_about_10min() -> None:
    t = scene_budget.resolve_target(depth="deep", target_minutes=None)
    assert 480 <= t.target_seconds <= 720


def test_target_minutes_overrides_depth() -> None:
    t = scene_budget.resolve_target(depth="brief", target_minutes=8)
    # target_minutes wins → 480 seconds with ±10% window.
    assert abs(t.target_seconds - 480.0) < 0.5
    assert t.depth == "brief"  # stored for telemetry, but seconds is set


def test_select_scenes_trims_to_max_window() -> None:
    # 10 paragraph scenes * 22s = 220s; brief max is 150s.
    scenes = [_scene("paragraph_card", duration=22.0, importance="low")
              for _ in range(10)]
    target = scene_budget.resolve_target(depth="brief", target_minutes=None)
    kept, report = scene_budget.select_scenes(scenes, target)
    assert report["estimated_seconds"] <= target.max_seconds
    assert report["scene_count"] < len(scenes)
    assert report["decisions"]  # at least one drop logged


def test_select_scenes_keeps_high_priority_first() -> None:
    scenes = [
        _scene("cover", duration=12.0, importance="high"),
        _scene("paragraph_card", duration=30.0, importance="low"),
        _scene("paragraph_card", duration=30.0, importance="low"),
        _scene("paragraph_card", duration=30.0, importance="low"),
        _scene("deadline_timeline", duration=28.0, importance="high"),
        _scene("penalty_table", duration=28.0, importance="high"),
        _scene("recap_card", duration=18.0, importance="medium"),
    ]
    target = scene_budget.resolve_target(depth="brief", target_minutes=None)
    kept, _ = scene_budget.select_scenes(scenes, target)
    kept_kinds = {s.scene_kind for s in kept}
    # Cover + recap + at least one high-priority factual should survive.
    assert "cover" in kept_kinds
    assert "recap_card" in kept_kinds
    assert ("deadline_timeline" in kept_kinds
            or "penalty_table" in kept_kinds)


def test_brief_standard_deep_change_scene_budget() -> None:
    scenes = [_scene("paragraph_card", duration=20.0) for _ in range(30)]
    brief = scene_budget.select_scenes(
        scenes, scene_budget.resolve_target(depth="brief", target_minutes=None),
    )[1]["scene_count"]
    standard = scene_budget.select_scenes(
        scenes, scene_budget.resolve_target(depth="standard", target_minutes=None),
    )[1]["scene_count"]
    deep = scene_budget.select_scenes(
        scenes, scene_budget.resolve_target(depth="deep", target_minutes=None),
    )[1]["scene_count"]
    assert brief < standard < deep
