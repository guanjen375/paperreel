from __future__ import annotations

from paperreel.audio.zh_tw_normalize import normalize_zh_tw_for_tts


def test_zh_tw_tts_normalizes_money_percent_ranges_and_years() -> None:
    text = (
        "NT$3,000 需於 45 天前繳清。"
        "取消級距為 89-60 天 30%、44~31 天 50%，2025 年 MSC FIT。"
    )
    out = normalize_zh_tw_for_tts(text)
    assert "新臺幣三千元" in out
    assert "四十五天" in out
    assert "八十九到六十天" in out
    assert "四十四到三十一天" in out
    assert "百分之三十" in out
    assert "百分之五十" in out
    assert "二零二五" in out
    assert "M S C" in out
    assert "F I T" in out


def test_zh_tw_tts_normalizes_plain_ntd_amounts() -> None:
    out = normalize_zh_tw_for_tts("改名費 3,000 元，取消收 100%。")
    assert "三千元" in out
    assert "百分之一百" in out
