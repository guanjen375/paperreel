"""Duration controller for sketchbook / document_explainer mode.

Maps a (depth, target_minutes) tuple to a concrete scene budget and
helps select / merge / split scenes so the final video lands within
±10% of the target whenever feasible.

The controller does NOT call the LLM — it operates on the candidate
scene list produced by the script writer. Priority rules:

- High-importance scenes (deadlines, money, percentages, risks,
  obligations) are kept first.
- Cover + recap_card always bookend the video.
- When over budget, low-importance paragraph_card / section_intro
  scenes are dropped first.
- When under budget, repeated high-importance scenes get
  paragraph_card siblings (one per scene) to add depth.

Cheap, deterministic, easy to reason about.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models import ScriptScene


# Approx seconds per scene by kind. Used to convert a target minute
# count into a scene count without having to TTS-render anything first.
_SECONDS_BY_KIND: dict[str, float] = {
    "cover": 12.0,
    "section_intro": 14.0,
    "deadline_timeline": 28.0,
    "penalty_table": 28.0,
    "checklist": 26.0,
    "risk_warning": 24.0,
    "do_dont": 22.0,
    "recap_card": 18.0,
    "paragraph_card": 22.0,
    "source_crop": 22.0,
    "key_number": 18.0,
}

_DEFAULT_SCENE_SECONDS = 22.0

# Importance ranks — higher number wins when culling.
_IMPORTANCE_RANK = {"high": 3, "medium": 2, "low": 1}
_KIND_PRIORITY = {
    "cover": 100,
    "recap_card": 95,
    "deadline_timeline": 90,
    "penalty_table": 88,
    "risk_warning": 85,
    "checklist": 80,
    "key_number": 75,
    "do_dont": 70,
    "source_crop": 60,
    "section_intro": 55,
    "paragraph_card": 40,
}


@dataclass(frozen=True)
class DurationTarget:
    depth: str
    target_seconds: float
    min_seconds: float
    max_seconds: float
    rationale: str


_DEPTH_TABLE: dict[str, tuple[float, float, float]] = {
    "brief":    (120.0,  90.0, 150.0),
    "standard": (300.0, 240.0, 360.0),
    "deep":     (600.0, 480.0, 720.0),
}


def resolve_target(*, depth: str | None, target_minutes: float | str | None
                   ) -> DurationTarget:
    """Resolve (depth, target_minutes) into a concrete seconds target.

    target_minutes overrides depth when present and parseable.
    """
    parsed_minutes: float | None = None
    if target_minutes is not None and not (
        isinstance(target_minutes, str) and target_minutes.lower() == "auto"
    ):
        try:
            parsed_minutes = float(target_minutes)
        except (TypeError, ValueError):
            parsed_minutes = None
    if parsed_minutes is not None and parsed_minutes > 0:
        seconds = parsed_minutes * 60.0
        return DurationTarget(
            depth=(depth or "standard").lower(),
            target_seconds=seconds,
            min_seconds=seconds * 0.9,
            max_seconds=seconds * 1.1,
            rationale=f"target_minutes override: {parsed_minutes:.1f} min",
        )
    key = (depth or "standard").lower()
    if key not in _DEPTH_TABLE:
        key = "standard"
    mid, lo, hi = _DEPTH_TABLE[key]
    return DurationTarget(
        depth=key,
        target_seconds=mid,
        min_seconds=lo,
        max_seconds=hi,
        rationale=f"depth={key} → {mid/60:.1f} min (±10% window {lo/60:.1f}–{hi/60:.1f})",
    )


def estimated_seconds(scene: ScriptScene) -> float:
    """Estimate how long ``scene`` will run; prefers the scene's own
    estimate, falls back to a kind-based table."""
    if scene.estimated_duration_sec and scene.estimated_duration_sec > 0:
        return float(scene.estimated_duration_sec)
    kind = scene.scene_kind or "paragraph_card"
    return _SECONDS_BY_KIND.get(kind, _DEFAULT_SCENE_SECONDS)


def _scene_score(scene: ScriptScene) -> float:
    """Higher = more important. Used to choose what to keep when over budget."""
    kind = (scene.scene_kind or "paragraph_card").lower()
    base = _KIND_PRIORITY.get(kind, 30.0)
    imp = (scene.importance or "").lower()
    base += _IMPORTANCE_RANK.get(imp, 0) * 5.0
    # Scenes that carry actual facts beat scenes that don't, even at
    # the same kind / importance.
    if scene.facts:
        base += min(10.0, len(scene.facts) * 2.0)
    if scene.evidence_spans:
        base += 5.0
    return base


def select_scenes(scenes: list[ScriptScene], target: DurationTarget
                  ) -> tuple[list[ScriptScene], dict]:
    """Trim or pad ``scenes`` to fit ``target``.

    Returns (selected_scenes, report). The report captures decisions
    so the run log can explain why something was dropped.
    """
    if not scenes:
        return [], {"reason": "no scenes provided"}

    keep = list(scenes)
    keep_seconds = sum(estimated_seconds(s) for s in keep)
    decisions: list[str] = []

    # Trim from low-priority side until we're at/under the max budget.
    while keep_seconds > target.max_seconds and len(keep) > 3:
        # Don't drop cover or recap_card — they bookend the video.
        candidates = [(i, s) for i, s in enumerate(keep)
                       if (s.scene_kind or "").lower()
                       not in ("cover", "recap_card")]
        if not candidates:
            break
        # Pick the lowest-priority scene in the middle.
        candidates.sort(key=lambda t: _scene_score(t[1]))
        i, sc = candidates[0]
        decisions.append(
            f"drop {sc.scene_id} ({sc.scene_kind}, score={_scene_score(sc):.1f})"
        )
        keep.pop(i)
        keep_seconds = sum(estimated_seconds(s) for s in keep)

    # Pad: when too short, duplicate paragraph cards behind the most
    # important factual scenes. We don't invent content here — we tell
    # the caller "expand these scenes". Padding is best-effort.
    pad_targets: list[str] = []
    if keep_seconds < target.min_seconds:
        sorted_by_score = sorted(keep, key=_scene_score, reverse=True)
        for sc in sorted_by_score:
            if (sc.scene_kind or "").lower() in ("cover", "recap_card",
                                                  "paragraph_card"):
                continue
            if keep_seconds >= target.min_seconds:
                break
            pad_targets.append(sc.scene_id)
            keep_seconds += _SECONDS_BY_KIND["paragraph_card"]
            decisions.append(f"pad after {sc.scene_id}")

    report = {
        "depth": target.depth,
        "target_seconds": round(target.target_seconds, 1),
        "min_seconds": round(target.min_seconds, 1),
        "max_seconds": round(target.max_seconds, 1),
        "estimated_seconds": round(keep_seconds, 1),
        "scene_count": len(keep),
        "rationale": target.rationale,
        "decisions": decisions,
        "pad_after_scene_ids": pad_targets,
    }
    return keep, report
