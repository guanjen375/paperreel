"""XTTS provider helpers that do not require loading the model."""
from __future__ import annotations

from paperreel.models import Scene, SceneStatus, VisualType
from paperreel.providers.tts_xtts import _split_xtts_text
from paperreel.stages.synthesize_audio import _audio_inputs


def test_xtts_zh_split_keeps_chunks_under_model_limit() -> None:
    text = (
        "請先記住這幾個關鍵時程：出發前八十九到六十天以上取消會收取百分之三十。"
        "接著是五十九到三十二天收取百分之五十；三十一到十六天收取百分之七十五。"
        "十五天內取消則收取百分之一百，請務必回到來源頁確認。"
    )
    chunks = _split_xtts_text(text, max_chars=80)
    assert len(chunks) >= 2
    assert all(0 < len(chunk) <= 80 for chunk in chunks)
    assert "".join(chunks) == text


def test_xtts_zh_split_hard_wraps_long_clause() -> None:
    text = "甲" * 205
    chunks = _split_xtts_text(text, max_chars=80)
    assert [len(chunk) for chunk in chunks] == [80, 80, 45]
    assert "".join(chunks) == text


def test_audio_manifest_schema_tracks_xtts_chunking() -> None:
    scene = Scene(
        scene_id="ch_001_sc_001",
        chapter_id="ch_001",
        title="測試",
        source_pages=[1],
        narration_text_zh_tw="這是一段測試旁白。",
        visual_type=VisualType.sketchbook_card,
        estimated_duration_sec=10.0,
        input_hash="hash",
        status=SceneStatus.pending,
    )
    inputs = _audio_inputs(scene, {"provider": "xtts", "sample_rate_hz": 24000})
    assert inputs["schema"] == "audio_artifact_v4"
    assert inputs["text_chunking"] == "xtts_zh_safe_80"
    assert inputs["normalizer"] == "zh_tw_tts_v1"
    assert inputs["tts_text_sha256"]
