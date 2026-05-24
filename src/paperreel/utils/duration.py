"""Duration estimator.

Spec:
    - target_minutes = clamp(round(cjk_char_count / auto_chars_per_minute),
                             auto_minutes_min, auto_minutes_max)
    - if page_count > 300 and heading_count > 15: allow 45..90 (i.e. raise floor)
    - script speed: ~230..260 CJK chars/min (default 240)
    - each scene 30..90s, recap every 6..10 min
"""
from __future__ import annotations

from dataclasses import dataclass


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class DurationPlan:
    target_minutes: float
    target_seconds: float
    chars_per_minute: float          # narration pace
    rationale: str


def estimate_target_minutes(
    *,
    cjk_char_count: int,
    page_count: int,
    heading_count: int,
    auto_chars_per_minute: float = 2200,
    auto_minutes_min: float = 3,
    auto_minutes_max: float = 120,
    user_target: float | str | None = "auto",
) -> DurationPlan:
    """Decide how long the video should be."""
    if isinstance(user_target, (int, float)) and not isinstance(user_target, bool):
        m = float(user_target)
        return DurationPlan(
            target_minutes=m,
            target_seconds=m * 60.0,
            chars_per_minute=240.0,
            rationale=f"user override: {m:.1f} min",
        )
    if isinstance(user_target, str) and user_target.lower() != "auto":
        try:
            m = float(user_target)
            return DurationPlan(
                target_minutes=m,
                target_seconds=m * 60.0,
                chars_per_minute=240.0,
                rationale=f"parsed override: {m:.1f} min",
            )
        except ValueError:
            pass

    raw = round(cjk_char_count / max(1.0, auto_chars_per_minute))
    minutes = clamp(float(raw), auto_minutes_min, auto_minutes_max)

    if page_count > 300 and heading_count > 15:
        # large structured book: bump up so we get meaningful coverage
        minutes = clamp(minutes, 45.0, 90.0)
        rationale = (
            f"auto: {cjk_char_count} CJK / {auto_chars_per_minute:.0f} ≈ {raw} min, "
            f"large book ({page_count}p / {heading_count} headings) -> clamp [45, 90] -> {minutes:.0f}"
        )
    else:
        rationale = (
            f"auto: {cjk_char_count} CJK / {auto_chars_per_minute:.0f} ≈ {raw} min, "
            f"clamp [{auto_minutes_min:.0f}, {auto_minutes_max:.0f}] -> {minutes:.0f}"
        )

    return DurationPlan(
        target_minutes=minutes,
        target_seconds=minutes * 60.0,
        chars_per_minute=240.0,
        rationale=rationale,
    )


def estimate_scene_seconds(narration_chars: int, *, chars_per_minute: float = 240.0) -> float:
    """Convert a narration's CJK char count to seconds at the configured rate.

    Adds 0.6s for pause / sentence boundaries, clamped to [12, 120]s.
    """
    base = (narration_chars / max(1.0, chars_per_minute)) * 60.0
    return clamp(base + 0.6, 12.0, 120.0)


def split_chars_per_scene(
    target_seconds: float,
    *,
    scene_min_sec: float = 30.0,
    scene_max_sec: float = 90.0,
    chars_per_minute: float = 240.0,
) -> tuple[int, int]:
    """Return (target chars per scene, suggested scene count)."""
    avg_scene = (scene_min_sec + scene_max_sec) / 2.0
    scene_count = max(1, round(target_seconds / avg_scene))
    seconds_per_scene = target_seconds / scene_count
    chars_per_scene = max(40, round(seconds_per_scene / 60.0 * chars_per_minute))
    return chars_per_scene, scene_count


def should_insert_recap(scene_index: int, total_scenes: int, *,
                        recap_every_minutes: float = 8.0,
                        avg_scene_seconds: float = 60.0) -> bool:
    if total_scenes == 0:
        return False
    scenes_between = max(1, round(recap_every_minutes * 60.0 / avg_scene_seconds))
    return scene_index > 0 and (scene_index % scenes_between == 0)
