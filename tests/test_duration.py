from __future__ import annotations

from pdf2lesson.utils.duration import (estimate_scene_seconds,
                                       estimate_target_minutes,
                                       should_insert_recap,
                                       split_chars_per_scene)


def test_short_pdf_clamps_to_lower_bound() -> None:
    plan = estimate_target_minutes(cjk_char_count=2000, page_count=10,
                                   heading_count=2)
    assert plan.target_minutes == 12.0


def test_huge_book_clamps_into_large_book_window() -> None:
    # Spec: page_count > 300 AND heading_count > 15 -> clamp [45, 90]
    plan = estimate_target_minutes(cjk_char_count=400_000, page_count=600,
                                   heading_count=30)
    assert plan.target_minutes == 90.0


def test_global_cap_120_when_not_large_book() -> None:
    # Heuristic would say 400_000/2200 ≈ 182 min, but heading_count too low
    # to enter the large-book bucket, so falls back to the global cap = 120.
    plan = estimate_target_minutes(cjk_char_count=400_000, page_count=150,
                                   heading_count=5)
    assert plan.target_minutes == 120.0


def test_about_60_minutes_for_130k_chars() -> None:
    # spec: ~130k CJK should hit ~60 min
    plan = estimate_target_minutes(cjk_char_count=130_000, page_count=400,
                                   heading_count=20)
    assert 45.0 <= plan.target_minutes <= 90.0


def test_user_override_passthrough() -> None:
    plan = estimate_target_minutes(cjk_char_count=200, page_count=1,
                                   heading_count=0, user_target=42)
    assert plan.target_minutes == 42.0
    assert plan.target_seconds == 42.0 * 60.0


def test_string_user_override() -> None:
    plan = estimate_target_minutes(cjk_char_count=200, page_count=1,
                                   heading_count=0, user_target="25")
    assert plan.target_minutes == 25.0


def test_estimate_scene_seconds_within_bounds() -> None:
    assert 12.0 <= estimate_scene_seconds(50) <= 120.0
    assert estimate_scene_seconds(2000) <= 120.0


def test_split_chars_per_scene_sane() -> None:
    chars, scenes = split_chars_per_scene(3600.0)  # 60-minute video
    assert chars > 0
    assert scenes >= 30


def test_recap_inserts_periodically() -> None:
    inserted = sum(1 for i in range(40)
                   if should_insert_recap(i, 40,
                                          recap_every_minutes=8.0,
                                          avg_scene_seconds=60.0))
    assert 2 <= inserted <= 8
