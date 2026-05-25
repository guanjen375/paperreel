"""Fact extraction regressions for contract-style PDFs."""
from __future__ import annotations

from paperreel.utils import fact_extract


def test_contract_cancellation_tiers_extract_as_penalty_rows() -> None:
    text = """
    甲方於郵輪出發前90 天以上取消者，乙方將收取全額訂金為取消費用。
    甲方於郵輪出發前89-60 天以上取消者，乙方將收取甲方全額船艙費用之30% 為取消費用。
    甲方於郵輪出發前59-32 天以上取消者，乙方將收取甲方全額船艙費用之50% 為取消費用。
    甲方於郵輪出發前31-16 天以上取消者，乙方將收取甲方全額船艙費用之75% 為取消費用。
    甲方於郵輪出發前15 天（含15 天內）取消者，乙方將收取甲方全額船艙費用之100% 為取消費用。
    出發前44~31 天甲方欲更改名單或艙房分配時，需付每人新台幣3,000 元改名手續費。
    出發前30 天內恕無法更動任何名字及艙房分配。
    """
    page_text = {1: text}
    page_facts = fact_extract.extract_from_pages(page_text)
    rows = fact_extract.group_for_scene_kind(
        "penalty_table", page_facts, page_text, max_items=8,
    )["rows"]
    blob = "\n".join(f"{r['condition']} -> {r['value']}" for r in rows)
    for token in ("全額訂", "30%", "50%", "75%", "100%", "3,000", "30 天內"):
        assert token in blob


def test_contract_deadline_ranges_extract_for_timeline() -> None:
    text = (
        "郵輪行程出發前45 天，甲方應繳付全額費用。"
        "甲方需於出發前45 天以上提供正確名單。"
        "出發前44~31 天更改名單需付費。"
        "出發前30 天內恕無法更動任何名字。"
    )
    page_text = {1: text}
    page_facts = fact_extract.extract_from_pages(page_text)
    events = fact_extract.group_for_scene_kind(
        "deadline_timeline", page_facts, page_text, max_items=8,
    )["events"]
    blob = "\n".join(e["value"] for e in events)
    assert "出發前45 天" in blob
    assert "出發前45 天以上" in blob
    assert "出發前44~31 天" in blob
    assert "出發前30 天內" in blob
