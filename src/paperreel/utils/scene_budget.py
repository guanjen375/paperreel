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
- When under budget, factual scenes get item-level grounded expansion
  siblings (penalty rows, risk items, checklist actions) to add depth.

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
_EXPANSION_SCENE_SECONDS = 18.0

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


def auto_target_minutes(*, page_count: int, cjk_char_count: int,
                        doc_kind: str | None, depth: str | None = None
                        ) -> tuple[float, str]:
    """Pick a useful explainer length when the user leaves target auto.

    This is intentionally conservative: auto should produce a focused
    overview, not a page-by-page reading. Explicit ``--target-minutes``
    remains the only normal user-facing length control.
    """
    pages = max(1, int(page_count or 1))
    chars = max(0, int(cjk_char_count or 0))
    kind = (doc_kind or "unknown").lower()
    depth_key = (depth or "standard").lower()

    # Dense contracts/policies need more time per page than slides, and
    # papers/reports benefit from a little more explanation around method
    # or metric context. The char term keeps long single-page forms from
    # being under-budgeted.
    if kind in {"contract", "policy", "form"}:
        raw = 1.4 + pages * 0.75 + chars / 3600.0
    elif kind == "paper":
        raw = 2.0 + pages * 0.55 + chars / 5000.0
    elif kind == "manual":
        raw = 1.8 + pages * 0.50 + chars / 5200.0
    elif kind == "report":
        raw = 2.0 + pages * 0.60 + chars / 4600.0
    elif kind == "slides":
        raw = 1.5 + pages * 0.28 + chars / 6500.0
    else:
        raw = 1.8 + pages * 0.45 + chars / 5600.0

    bounds = {
        "brief": (2.0, 3.0),
        "standard": (2.5, 6.0),
        "deep": (5.0, 10.0),
    }.get(depth_key, (2.5, 6.0))
    minutes = max(bounds[0], min(bounds[1], raw))
    # Round to the nearest half-minute so CLI output stays readable.
    minutes = round(minutes * 2.0) / 2.0
    rationale = (
        f"auto explainer target: doc_kind={kind}, pages={pages}, "
        f"chars={chars}, depth={depth_key} -> {minutes:.1f} min"
    )
    return minutes, rationale


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


def _grounded_item_count(items: list[dict]) -> int:
    seen: set[tuple[int, str, str, str]] = set()
    for it in items:
        if not isinstance(it, dict) or not isinstance(it.get("page"), int):
            continue
        key = (
            it["page"],
            str(it.get("text") or it.get("context") or it.get("label") or ""),
            str(it.get("condition") or ""),
            str(it.get("value") or ""),
        )
        seen.add(key)
    return len(seen)


def _expansion_capacity(scene: ScriptScene) -> int:
    """How many distinct grounded expansion cards this scene can support."""
    kind = (scene.scene_kind or "paragraph_card").lower()
    payload = scene.layout_payload or {}
    if kind == "penalty_table":
        return _grounded_item_count(list(payload.get("rows") or []))
    if kind == "risk_warning":
        return _grounded_item_count(list(payload.get("items") or []))
    if kind == "checklist":
        return _grounded_item_count(list(payload.get("items") or []))
    if kind == "deadline_timeline":
        return _grounded_item_count(list(payload.get("events") or []))
    if kind == "do_dont":
        return _grounded_item_count(
            list(payload.get("do") or []) + list(payload.get("dont") or [])
        )
    if kind == "key_number":
        return _grounded_item_count(list(payload.get("items") or []))
    return 0


def _can_expand(scene: ScriptScene) -> bool:
    """True when a scene has distinct grounded items worth expanding."""
    return _expansion_capacity(scene) > 0


def _expansion_slots(scenes: list[ScriptScene]) -> list[ScriptScene]:
    """Return scene ids in round-robin item order by importance.

    Round-robin keeps a high-scoring timeline from consuming every
    available expansion before a penalty table or risk card gets a turn.
    """
    ranked = [sc for sc in sorted(scenes, key=_scene_score, reverse=True) if _can_expand(sc)]
    max_capacity = max((_expansion_capacity(sc) for sc in ranked), default=0)
    slots: list[ScriptScene] = []
    for item_index in range(max_capacity):
        for sc in ranked:
            if item_index < _expansion_capacity(sc):
                slots.append(sc)
    return slots


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

    # Pad: when too short, request grounded expansion cards behind the
    # most important factual scenes. We don't invent content here — we
    # tell the caller "expand these scenes". The caller inserts distinct
    # item-level cards that reuse the original scene's facts/evidence.
    pad_targets: list[str] = []
    if keep_seconds < target.min_seconds:
        slots = _expansion_slots(keep)
        padding_goal = min(target.target_seconds, target.max_seconds)
        for sc in slots:
            if keep_seconds >= padding_goal:
                break
            pad_targets.append(sc.scene_id)
            keep_seconds += _EXPANSION_SCENE_SECONDS
            decisions.append(f"pad after {sc.scene_id}")
        if keep_seconds < target.min_seconds:
            decisions.append(
                "under target: not enough distinct grounded facts/evidence "
                "to expand without repeating content"
            )

    report = {
        "depth": target.depth,
        "target_seconds": round(target.target_seconds, 1),
        "min_seconds": round(target.min_seconds, 1),
        "max_seconds": round(target.max_seconds, 1),
        "estimated_seconds": round(keep_seconds, 1),
        "scene_count": len(keep) + len(pad_targets),
        "base_scene_count": len(keep),
        "expansion_scene_count": len(pad_targets),
        "expansion_capacity": len(_expansion_slots(keep)),
        "rationale": target.rationale,
        "decisions": decisions,
        "pad_after_scene_ids": pad_targets,
    }
    return keep, report
