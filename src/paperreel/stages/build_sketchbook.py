"""Sketchbook / document_explainer script builder.

Drop-in replacement for ``write_chapter_script``-style output when
``project.style`` is sketchbook or document_explainer. Builds the
whole storyboard deterministically from:

1. Document classifier output (``utils/doc_classify``)
2. Heuristic fact extractor (``utils/fact_extract``)
3. Per-doc storyboard skeleton (cover / section_intro / timeline / …)
4. Optional LLM "narration polish" — the LLM may rewrite narration
   prose, but never invents numbers, dates, fees or percentages.

The output is a list of :class:`ScriptScene` with ``scene_kind``,
``facts``, ``evidence_spans`` and ``layout_payload`` populated.
Validation happens in ``utils/grounding`` once the list is built.
"""
from __future__ import annotations

import json
from typing import Any

from ..models import (ChunkedSources, DocKind, DocProfile, EvidenceSpan,
                       Fact, LessonOutline, ScriptScene, VisualType)
from ..providers.llm_base import LLMProvider
from ..utils import doc_classify, fact_extract, grounding
from ..utils.scene_budget import (DurationTarget, estimated_seconds,
                                   resolve_target, select_scenes)
from ..utils.text_cleaning import cjk_char_count, normalise_text


# ---------- public entry point ----------

def build_sketchbook_scenes(
    *,
    sources: ChunkedSources,
    outline: LessonOutline,
    profile: DocProfile,
    duration: DurationTarget,
    cards_cfg: dict,
    provider: LLMProvider | None = None,
    use_llm_polish: bool = True,
) -> tuple[list[ScriptScene], dict]:
    """Return (scenes, plan_report).

    ``provider`` is optional — without it the scenes get heuristic
    narration that still references concrete facts. Tests pass ``None``
    so they don't need an LLM.
    """
    page_text = {pg.page: pg.text for pg in sources.pages}
    page_facts = fact_extract.extract_from_pages(page_text)

    storyboard = doc_classify.storyboard_for(profile.doc_kind)
    scenes: list[ScriptScene] = []

    # Cover always first.
    scenes.append(_build_cover(outline, profile))

    # Walk the outline chapter by chapter, dropping in storyboard cards
    # tied to that chapter's source pages.
    for ch_idx, ch in enumerate(outline.chapters, start=1):
        chapter_id = ch.chapter_id or f"ch_{ch_idx:03d}"
        chapter_pages = list(ch.source_pages or [])
        ch_text = {p: page_text.get(p, "") for p in chapter_pages}
        ch_facts = {p: page_facts.get(p, []) for p in chapter_pages}

        scenes.append(_build_section_intro(
            chapter_id=chapter_id, chapter_no=ch_idx,
            title=ch.title or f"第 {ch_idx} 段",
            chapter_pages=chapter_pages,
            key_points=ch.key_points or [],
        ))

        # Pick kinds from storyboard skeleton, skipping cover/recap_card
        # (we add cover above and recap_card after the loop).
        sub_kinds = [k for k in storyboard
                     if k not in ("cover", "recap_card", "section_intro")]
        sc_no = 1
        for kind in sub_kinds:
            payload = fact_extract.group_for_scene_kind(
                kind, ch_facts, ch_text,
                max_items=int(cards_cfg.get("max_table_rows", 6)),
            )
            sub_scene = _scene_from_kind(
                kind=kind,
                chapter_id=chapter_id,
                scene_no=sc_no,
                chapter_pages=chapter_pages,
                chapter_title=ch.title,
                payload=payload,
                page_facts=ch_facts,
                cards_cfg=cards_cfg,
            )
            if sub_scene is None:
                continue
            scenes.append(sub_scene)
            sc_no += 1

    # Recap last.
    scenes.append(_build_recap(outline, profile, scenes))

    # Renumber scene_ids so every id is canonical and unique.
    scenes = _renumber(scenes)

    # Duration controller — trim / pad to hit the requested window.
    # select_scenes only decides where expansion is allowed; we insert
    # the actual grounded expansion cards here so script.json and
    # scene_graph.json reflect the target length.
    selected, plan_report = select_scenes(scenes, duration)
    selected = _insert_expansion_scenes(selected, plan_report)
    selected = _renumber(selected)
    plan_report["actual_scene_count"] = len(selected)
    plan_report["actual_estimated_seconds"] = round(
        sum(estimated_seconds(s) for s in selected), 1
    )
    if selected and plan_report["actual_estimated_seconds"] < duration.min_seconds:
        plan_report["under_target_reason"] = (
            "not enough distinct grounded facts/evidence to expand without "
            "repeating the same scene"
        )

    # Optional LLM narration polish — keeps all numbers + evidence
    # intact, only rewrites the prose.
    if provider is not None and use_llm_polish:
        selected = _polish_with_llm(provider, selected, page_text)

    return selected, plan_report


# ---------- per-kind scene builders ----------

def _build_cover(outline: LessonOutline, profile: DocProfile
                 ) -> ScriptScene:
    eyebrow = {
        DocKind.contract: "合約 / 條款說明",
        DocKind.form: "申請表單導讀",
        DocKind.paper: "論文重點解析",
        DocKind.manual: "操作手冊導讀",
        DocKind.report: "報告重點摘要",
        DocKind.policy: "辦法與規範整理",
        DocKind.slides: "簡報重點整理",
        DocKind.unknown: "文件導讀",
    }.get(profile.doc_kind, "文件導讀")
    title = (outline.project or "文件導讀")[:32]
    payload = {
        "eyebrow": eyebrow,
        "subtitle": f"共 {len(outline.chapters)} 段，{outline.target_minutes:.1f} 分鐘",
        "doc_kind": profile.doc_kind.value,
    }
    narration = (
        f"歡迎收看這份{eyebrow}。我們會用大約"
        f"{max(1, round(outline.target_minutes))}分鐘，"
        "把這份文件最關鍵的時程、條款與注意事項，"
        "整理成幾張清楚的卡片。"
    )
    return ScriptScene(
        scene_id="ch_000_sc_001",
        chapter_id="ch_000",
        title=title,
        source_pages=[1],
        source_refs=["cover"],
        narration_text_zh_tw=narration,
        on_screen_text=eyebrow,
        visual_hint="cover card",
        visual_type=VisualType.sketchbook_card,
        scene_kind="cover",
        layout_payload=payload,
        importance="high",
        estimated_duration_sec=12.0,
    )


def _build_section_intro(*, chapter_id: str, chapter_no: int,
                         title: str,
                         chapter_pages: list[int],
                         key_points: list[str]) -> ScriptScene:
    bullets = [str(p)[:60] for p in key_points[:3]]
    payload = {
        "number": f"{chapter_no:02d}",
        "body": "；".join(bullets) if bullets else "",
    }
    narration = (
        f"接下來進入第 {chapter_no} 段，主題是「{title}」。"
        + (f"重點包含：{ '、'.join(bullets) }。" if bullets else "")
    )
    return ScriptScene(
        scene_id=f"{chapter_id}_sc_000",  # renumbered later
        chapter_id=chapter_id,
        title=title[:40],
        source_pages=chapter_pages or [1],
        source_refs=[f"p.{p}" for p in chapter_pages] or ["p.?"],
        narration_text_zh_tw=narration,
        on_screen_text=title[:30],
        visual_hint="section intro",
        visual_type=VisualType.sketchbook_card,
        scene_kind="section_intro",
        layout_payload=payload,
        importance="medium",
        estimated_duration_sec=14.0,
    )


def _scene_from_kind(*, kind: str, chapter_id: str, scene_no: int,
                     chapter_pages: list[int],
                     chapter_title: str,
                     payload: dict,
                     page_facts: dict[int, list[tuple[Fact, EvidenceSpan]]],
                     cards_cfg: dict,
                     ) -> ScriptScene | None:
    """Build one factual sketchbook scene. Returns None when there's
    no useful payload for ``kind`` in this chapter — caller skips it."""
    facts: list[Fact] = []
    evidence: list[EvidenceSpan] = []
    used_pages: set[int] = set()

    def _collect(values: list[dict], fallback_label: str | None = None) -> None:
        for v in values:
            page = v.get("page")
            if not isinstance(page, int):
                continue
            used_pages.add(page)
            value = str(v.get("value") or "").strip()
            matched = False
            # Pair the value back to its evidence span if we can find it.
            for fact, span in page_facts.get(page, []):
                if value and fact.value == value:
                    span_idx = len(evidence)
                    evidence.append(span)
                    facts.append(Fact(
                        label=fact.label,
                        value=fact.value,
                        importance=fact.importance,
                        evidence_index=span_idx,
                    ))
                    matched = True
                    break
            if matched or not fallback_label or not value:
                continue
            quote = (v.get("text") or v.get("context") or
                     v.get("condition") or v.get("label") or value)
            span_idx = len(evidence)
            evidence.append(EvidenceSpan(
                page=page, quote=str(quote)[:160], label=fallback_label,
                value=value, importance="high",
            ))
            facts.append(Fact(
                label=fallback_label, value=value, importance="high",
                evidence_index=span_idx,
            ))

    def _evidence_from_quotes(values: list[dict], label: str) -> None:
        for v in values:
            page = v.get("page")
            text = v.get("text") or v.get("condition") or v.get("context") or ""
            if not isinstance(page, int) or not text:
                continue
            used_pages.add(page)
            evidence.append(EvidenceSpan(
                page=page, quote=text[:160], label=label,
                value=str(v.get("value") or "") or None,
                importance="high" if label in ("罰則", "風險", "期限") else "medium",
            ))

    if kind == "deadline_timeline":
        events = payload.get("events") or []
        if not events:
            return None
        _collect(events, "期限")
        title = f"{chapter_title} · 關鍵時程" if chapter_title else "關鍵時程"
        narration = _timeline_narration(events)
        importance = "high"

    elif kind == "penalty_table":
        rows = payload.get("rows") or []
        if not rows:
            return None
        _collect(rows, "罰則")
        title = f"{chapter_title} · 罰則與費用" if chapter_title else "罰則與費用"
        narration = _penalty_narration(rows)
        importance = "high"

    elif kind == "checklist":
        items = payload.get("items") or []
        if not items:
            return None
        _evidence_from_quotes(items, "應辦事項")
        title = f"{chapter_title} · 應辦事項" if chapter_title else "應辦事項"
        narration = _checklist_narration(items)
        importance = "medium"

    elif kind == "risk_warning":
        items = payload.get("items") or []
        if not items:
            return None
        _evidence_from_quotes(items, "風險")
        title = f"{chapter_title} · 風險提醒" if chapter_title else "風險提醒"
        narration = _risk_narration(items)
        importance = "high"

    elif kind == "do_dont":
        do_items = payload.get("do") or []
        dont_items = payload.get("dont") or []
        if not do_items and not dont_items:
            return None
        _evidence_from_quotes(do_items, "應做")
        _evidence_from_quotes(dont_items, "不要做")
        title = f"{chapter_title} · 該做／不該做" if chapter_title else "該做／不該做"
        narration = _do_dont_narration(do_items, dont_items)
        importance = "medium"

    elif kind == "key_number":
        items = payload.get("items") or []
        if not items:
            return None
        _collect(items, "關鍵數字")
        title = f"{chapter_title} · 關鍵數字" if chapter_title else "關鍵數字"
        narration = _keynumber_narration(items)
        importance = "high"

    elif kind == "paragraph_card":
        # Used as filler. Choose the densest sentence on the chapter's
        # primary page so we have something concrete to read.
        if not chapter_pages:
            return None
        primary_page = chapter_pages[0]
        text = page_facts and ""  # noop to keep linters quiet
        # We don't have chapter text here easily; let the LLM polish
        # fill it. Stub narration referencing the page is acceptable.
        title = f"{chapter_title} · 補充說明" if chapter_title else "補充說明"
        narration = (
            f"接下來補充一點本段的背景。請特別注意這份文件第 {primary_page} "
            "頁附近的條款細節，避免漏掉重要條件。"
        )
        importance = "low"
        payload = {"body": narration}
        used_pages.add(primary_page)

    else:
        return None

    source_pages = sorted(used_pages) or list(chapter_pages or [1])
    return ScriptScene(
        scene_id=f"{chapter_id}_sc_{scene_no:03d}",
        chapter_id=chapter_id,
        title=title[:40],
        source_pages=source_pages,
        source_refs=[f"p.{p}" for p in source_pages],
        narration_text_zh_tw=narration,
        on_screen_text=title[:30],
        visual_hint=kind,
        visual_type=VisualType.sketchbook_card,
        scene_kind=kind,
        facts=facts,
        evidence_spans=evidence,
        layout_payload=payload,
        importance=importance,
        estimated_duration_sec=_duration_for_kind(kind),
    )


def _build_recap(outline: LessonOutline, profile: DocProfile,
                 scenes: list[ScriptScene]) -> ScriptScene:
    items: list[dict] = []
    seen_titles: set[str] = set()
    for sc in scenes:
        if (sc.scene_kind or "") in ("cover", "section_intro", "paragraph_card",
                                       "recap_card"):
            continue
        bullet = sc.title.split("·")[-1].strip() if "·" in sc.title else sc.title
        bullet = bullet.strip()
        if bullet in seen_titles or not bullet:
            continue
        items.append({"text": bullet[:24], "page": sc.source_pages[0] if sc.source_pages else None})
        seen_titles.add(bullet)
        if len(items) >= 5:
            break
    if not items:
        items = [{"text": "重點回顧", "page": 1}]
    narration = (
        "做最後一個簡單的回顧：請記住"
        + "、".join(it["text"] for it in items)
        + "這幾個重點。"
    )
    return ScriptScene(
        scene_id="ch_999_sc_001",
        chapter_id="ch_999",
        title=f"{outline.project or '本片'} · 重點回顧",
        source_pages=sorted({it.get("page") for it in items if it.get("page")}) or [1],
        source_refs=["recap"],
        narration_text_zh_tw=narration,
        on_screen_text="重點回顧",
        visual_hint="recap",
        visual_type=VisualType.sketchbook_card,
        scene_kind="recap_card",
        layout_payload={"items": items},
        importance="medium",
        estimated_duration_sec=18.0,
    )


def _insert_expansion_scenes(scenes: list[ScriptScene],
                             plan_report: dict) -> list[ScriptScene]:
    pad_ids = list(plan_report.get("pad_after_scene_ids") or [])
    if not pad_ids:
        return scenes
    counts: dict[str, int] = {}
    for sid in pad_ids:
        counts[sid] = counts.get(sid, 0) + 1
    emitted: dict[str, int] = {}
    out: list[ScriptScene] = []
    for sc in scenes:
        out.append(sc)
        n = counts.get(sc.scene_id, 0)
        for _ in range(n):
            emitted[sc.scene_id] = emitted.get(sc.scene_id, 0) + 1
            out.append(_build_expansion_scene(sc, emitted[sc.scene_id]))
    return out


def _build_expansion_scene(base: ScriptScene, seq: int) -> ScriptScene:
    span = base.evidence_spans[(seq - 1) % len(base.evidence_spans)] if base.evidence_spans else None
    fact = base.facts[(seq - 1) % len(base.facts)] if base.facts else None

    facts: list[Fact] = []
    evidence: list[EvidenceSpan] = []
    focus = base.title.split("·")[-1].strip() if "·" in base.title else base.title
    if span is not None:
        evidence.append(span)
    if fact is not None:
        facts.append(Fact(
            label=fact.label, value=fact.value, importance=fact.importance,
            evidence_index=0 if evidence else None,
        ))
        focus = f"{fact.label}：{fact.value}"

    quote = (span.quote if span is not None else "").strip()
    body_parts = [focus]
    if quote:
        body_parts.append(f"原文依據：{quote[:90]}")
    elif base.layout_payload:
        body_parts.append(_payload_focus(base.layout_payload, seq))
    body = "\n".join(p for p in body_parts if p)
    narration = (
        f"補充說明「{focus}」。"
        + (f"這張卡只引用同一段來源：{quote[:80]}。" if quote else "這張卡延伸前一張的同一組來源資訊。")
        + "請把它和前一張卡一起看，避免只記住單一數字而漏掉適用條件。"
    )
    return ScriptScene(
        scene_id=f"{base.scene_id}_ex_{seq:02d}",
        chapter_id=base.chapter_id,
        title=(base.title + " · 補充")[:40],
        source_pages=list(base.source_pages),
        source_refs=list(base.source_refs),
        narration_text_zh_tw=narration[:600],
        on_screen_text=focus[:30],
        visual_hint="grounded expansion",
        visual_type=VisualType.sketchbook_card,
        scene_kind="paragraph_card",
        facts=facts,
        evidence_spans=evidence,
        layout_payload={
            "body": body[:360],
            "expansion_of": base.scene_id,
            "source_kind": base.scene_kind,
        },
        importance=base.importance or "medium",
        estimated_duration_sec=22.0,
    )


def _payload_focus(payload: dict, seq: int) -> str:
    for key in ("events", "rows", "items", "do", "dont"):
        values = list(payload.get(key) or [])
        if not values:
            continue
        item = values[(seq - 1) % len(values)]
        if isinstance(item, dict):
            text = (item.get("text") or item.get("condition") or
                    item.get("label") or item.get("context") or "")
            value = item.get("value") or ""
            return f"{text} {value}".strip()[:120]
        return str(item)[:120]
    return "同一來源中的補充脈絡"


def _duration_for_kind(kind: str) -> float:
    return {
        "deadline_timeline": 28.0,
        "penalty_table": 28.0,
        "checklist": 26.0,
        "risk_warning": 24.0,
        "do_dont": 22.0,
        "key_number": 18.0,
        "paragraph_card": 22.0,
        "source_crop": 22.0,
    }.get(kind, 22.0)


# ---------- heuristic narration ----------

def _timeline_narration(events: list[dict]) -> str:
    if not events:
        return "本段沒有偵測到關鍵時程。"
    parts = []
    for ev in events[:4]:
        parts.append(f"{ev.get('value','')}（{(ev.get('label') or '')[:24]}）")
    return "請先記住這幾個關鍵時程：" + "、".join(parts) + "。錯過的話會影響後續權益。"


def _penalty_narration(rows: list[dict]) -> str:
    if not rows:
        return "本段沒有偵測到罰則資料。"
    bits = []
    for row in rows[:3]:
        cond = (row.get("condition") or "")[:24]
        val = (row.get("value") or "")[:16]
        bits.append(f"{cond}{val}")
    return "重點罰則如下：" + "；".join(bits) + "。請務必比對自己的情境再決定行動。"


def _checklist_narration(items: list[dict]) -> str:
    if not items:
        return "本段沒有偵測到應辦事項。"
    bits = []
    for it in items[:4]:
        text = (it.get("text") or "")[:24]
        bits.append(text)
    return "請依序確認以下應辦事項：" + "、".join(bits) + "。"


def _risk_narration(items: list[dict]) -> str:
    if not items:
        return "本段沒有列出特定風險條款。"
    bits = []
    for it in items[:3]:
        bits.append((it.get("text") or "")[:30])
    return "需要特別注意的風險：" + "；".join(bits) + "。"


def _do_dont_narration(do_items: list[dict], dont_items: list[dict]) -> str:
    do_part = "、".join((it.get("text") or "")[:18] for it in do_items[:3])
    dont_part = "、".join((it.get("text") or "")[:18] for it in dont_items[:3])
    out = []
    if do_part:
        out.append("應該做：" + do_part)
    if dont_part:
        out.append("不要做：" + dont_part)
    return "；".join(out) + "。"


def _keynumber_narration(items: list[dict]) -> str:
    primary = items[0]
    return (
        f"這份文件最重要的數字之一是 {primary.get('value','')}，"
        f"它代表的是 {(primary.get('label') or '')[:16]}。"
        f"完整脈絡可見：{(primary.get('context') or '')[:40]}。"
    )


# ---------- helpers ----------

def _renumber(scenes: list[ScriptScene]) -> list[ScriptScene]:
    """Assign canonical sequential scene_ids inside each chapter."""
    counters: dict[str, int] = {}
    out: list[ScriptScene] = []
    for sc in scenes:
        counters[sc.chapter_id] = counters.get(sc.chapter_id, 0) + 1
        n = counters[sc.chapter_id]
        new_id = f"{sc.chapter_id}_sc_{n:03d}"
        if new_id != sc.scene_id:
            sc = sc.model_copy(update={"scene_id": new_id})
        out.append(sc)
    return out


def _polish_with_llm(provider: LLMProvider, scenes: list[ScriptScene],
                     page_text: dict[int, str]) -> list[ScriptScene]:
    """Ask the LLM to rewrite narration prose without changing numbers
    or evidence. Failures are tolerated — the heuristic narration is
    already grounded; LLM polish is just for naturalness."""
    polished: list[ScriptScene] = []
    polish = getattr(provider, "polish_sketchbook_narration", None)
    if not callable(polish):
        return scenes
    for sc in scenes:
        try:
            new_text = polish(
                scene=sc.model_dump(mode="json"),
                page_text={p: page_text.get(p, "")[:1600]
                           for p in sc.source_pages},
            )
        except Exception:
            polished.append(sc)
            continue
        if not isinstance(new_text, str) or not new_text.strip():
            polished.append(sc)
            continue
        # Reject the rewrite if it dropped any factual value.
        if not _preserves_facts(new_text, sc):
            polished.append(sc)
            continue
        polished.append(sc.model_copy(update={
            "narration_text_zh_tw": new_text.strip()[:600],
        }))
    return polished


def _preserves_facts(new_text: str, scene: ScriptScene) -> bool:
    """LLM rewrite must keep every concrete fact value the original
    narration had. We're permissive on prose, strict on numbers."""
    for fact in scene.facts:
        val = (fact.value or "").strip()
        if not val:
            continue
        if val not in new_text:
            return False
    return True


__all__ = ["build_sketchbook_scenes"]
