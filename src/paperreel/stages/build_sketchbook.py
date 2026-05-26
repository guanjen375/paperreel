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

import hashlib
import json
from typing import Any

from ..models import (ChunkedSources, DocKind, DocProfile, EvidenceSpan,
                       Fact, LessonOutline, ScriptScene, VisualCandidate,
                       VisualType)
from ..providers.llm_base import LLMProvider
from ..utils import doc_classify, fact_extract, grounding
from ..utils.visual_inventory import (anchor_from_candidate,
                                      candidate_source_path,
                                      useful_candidates)
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

    if _is_visual_first_profile(profile):
        return _build_visual_first_scenes(
            sources=sources, outline=outline, profile=profile,
            duration=duration, provider=provider,
            use_llm_polish=use_llm_polish, page_text=page_text,
        )

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


# ---------- visual-first storyboard ----------


def _is_visual_first_profile(profile: DocProfile) -> bool:
    return bool(
        profile.document_visual_rich
        and profile.doc_kind not in {DocKind.contract, DocKind.form, DocKind.policy}
    )


def _build_visual_first_scenes(
    *,
    sources: ChunkedSources,
    outline: LessonOutline,
    profile: DocProfile,
    duration: DurationTarget,
    provider: LLMProvider | None,
    use_llm_polish: bool,
    page_text: dict[int, str],
) -> tuple[list[ScriptScene], dict]:
    candidates = useful_candidates(list(sources.visual_inventory or []))
    if not candidates:
        # Extremely old artefacts can be classified visual-rich from the
        # image count but lack inventory details. Fall back to the
        # existing text-card builder rather than invent visuals.
        return build_sketchbook_scenes(
            sources=sources,
            outline=outline,
            profile=profile.model_copy(update={"document_visual_rich": False}),
            duration=duration,
            cards_cfg={},
            provider=provider,
            use_llm_polish=use_llm_polish,
        )

    scenes: list[ScriptScene] = [_build_cover(outline, profile)]
    scenes.append(_build_visual_walkthrough_intro(outline, sources))

    selected_candidates = _select_visual_candidates(candidates, sources, duration)
    idx = 0
    scene_no = 1
    while idx < len(selected_candidates):
        cand = selected_candidates[idx]
        next_cand = selected_candidates[idx + 1] if idx + 1 < len(selected_candidates) else None
        if next_cand is not None and _should_build_comparison(cand, next_cand):
            scenes.append(_build_comparison_visual_scene(
                left=cand, right=next_cand, sources=sources, scene_no=scene_no,
            ))
            idx += 2
        else:
            scenes.append(_build_single_visual_scene(
                candidate=cand, sources=sources, scene_no=scene_no,
            ))
            idx += 1
        scene_no += 1

    scenes.append(_build_recap(outline, profile, scenes))
    scenes = _renumber(scenes)
    selected, plan_report = select_scenes(scenes, duration)
    selected = _renumber(selected)
    plan_report.update({
        "visual_first": True,
        "source_visual_candidate_count": len(candidates),
        "selected_source_visual_count": sum(
            1 for sc in selected if sc.visual_anchor is not None
        ),
        "actual_scene_count": len(selected),
        "actual_estimated_seconds": round(
            sum(estimated_seconds(s) for s in selected), 1
        ),
    })
    if provider is not None and use_llm_polish:
        selected = _polish_with_llm(provider, selected, page_text)
    return selected, plan_report


def _build_visual_walkthrough_intro(outline: LessonOutline,
                                    sources: ChunkedSources) -> ScriptScene:
    visual_count = sum(1 for c in sources.visual_inventory if c.likely_useful)
    title = "先看來源視覺"
    body = f"挑出 {visual_count} 個來源圖片、圖表或截圖候選，接著只看最能說明概念的部分。"
    return ScriptScene(
        scene_id="ch_000_sc_002",
        chapter_id="ch_000",
        title=title,
        source_pages=[1],
        source_refs=["p.1"],
        narration_text_zh_tw=(
            "這份文件不是只適合做文字摘要。接下來我們會把重點放在來源畫面："
            "先看圖、表、截圖或範例，再用旁白解釋它真正要你理解的概念。"
        ),
        on_screen_text="看來源，不背卡片",
        visual_hint="visual walkthrough intro",
        visual_type=VisualType.sketchbook_card,
        scene_kind="section_intro",
        layout_payload={"number": "00", "body": body},
        importance="medium",
        estimated_duration_sec=14.0,
    )


def _select_visual_candidates(candidates: list[VisualCandidate],
                              sources: ChunkedSources,
                              duration: DurationTarget) -> list[VisualCandidate]:
    desired = max(4, min(len(candidates), int(max(1.0, duration.target_seconds - 44.0) // 27.0)))
    page_by_no = {p.page: p for p in sources.pages}
    ordered = sorted(candidates, key=lambda c: (c.page, -c.salience_score, c.candidate_id))
    selected: list[VisualCandidate] = []
    seen_topic_keys: set[str] = set()
    seen_pages: set[int] = set()
    for cand in ordered:
        topic = _visual_topic_title(cand, page_by_no)
        key = topic[:18]
        if cand.page in seen_pages:
            continue
        if key in seen_topic_keys and len(selected) >= desired // 2:
            continue
        selected.append(cand)
        seen_pages.add(cand.page)
        seen_topic_keys.add(key)
        if len(selected) >= desired:
            return selected
    for cand in ordered:
        if cand in selected:
            continue
        selected.append(cand)
        if len(selected) >= desired:
            break
    return selected


def _visual_topic_title(candidate: VisualCandidate,
                        page_by_no: dict[int, Any]) -> str:
    if candidate.nearby_heading:
        return str(candidate.nearby_heading)[:36]
    if candidate.nearby_caption:
        return str(candidate.nearby_caption)[:36]
    page = page_by_no.get(candidate.page)
    if page is not None:
        for heading in getattr(page, "headings", []) or []:
            h = str(heading).strip()
            if h:
                return h[:36]
        for line in str(getattr(page, "text", "")).splitlines():
            s = line.strip()
            if 4 <= len(s) <= 60:
                return s[:36]
    return f"第 {candidate.page} 頁視覺重點"


def _scene_kind_for_candidate(candidate: VisualCandidate) -> str:
    role = (candidate.visual_role or "unknown").lower()
    blob = " ".join([
        candidate.nearby_heading or "",
        candidate.nearby_caption or "",
        candidate.nearby_text or "",
    ]).lower()
    if role == "source_table":
        return "source_table_explainer"
    if role == "source_screenshot":
        return "source_screenshot_explainer"
    if any(term in blob for term in ("步驟", "流程", "三部曲", "設定", "調整", "workflow", "step")):
        return "process_visual_card"
    if role in {"source_diagram", "source_chart"}:
        return "figure_explainer"
    return "source_visual_explainer"


def _screen_callouts(candidate: VisualCandidate) -> list[str]:
    text = " ".join([
        candidate.nearby_heading or "",
        candidate.nearby_caption or "",
        candidate.nearby_text or "",
    ])
    role = candidate.visual_role or "source_visual"
    callouts: list[str] = []
    if role == "source_table":
        callouts.extend(["先看欄位", "找關鍵列"])
    elif role == "source_screenshot":
        callouts.extend(["看按鈕位置", "留意設定值"])
    elif role == "source_chart":
        callouts.extend(["看分布方向", "找異常區段"])
    elif role == "source_diagram":
        callouts.extend(["先看流程", "再看因果"])
    else:
        callouts.extend(["觀察主體", "注意差異"])
    if "光圈" in text or "景深" in text:
        callouts.append("景深變化")
    if "快門" in text:
        callouts.append("動作感")
    if "ISO" in text or "感光度" in text:
        callouts.append("雜訊與亮度")
    if "白平衡" in text:
        callouts.append("色溫偏移")
    return callouts[:4]


def _source_evidence(candidate: VisualCandidate) -> list[EvidenceSpan]:
    if not candidate.source_quote:
        return []
    return [EvidenceSpan(
        page=candidate.page,
        quote=str(candidate.source_quote)[:160],
        label="來源視覺脈絡",
        importance="medium",
    )]


def _build_single_visual_scene(*, candidate: VisualCandidate,
                               sources: ChunkedSources,
                               scene_no: int) -> ScriptScene:
    page_by_no = {p.page: p for p in sources.pages}
    topic = _visual_topic_title(candidate, page_by_no)
    kind = _scene_kind_for_candidate(candidate)
    anchor = anchor_from_candidate(candidate, why=f"用第 {candidate.page} 頁來源視覺說明「{topic}」")
    source_path = candidate_source_path(candidate)
    callouts = _screen_callouts(candidate)
    headline = _short_headline(topic)
    payload = {
        "headline": headline,
        "callouts": [{"text": c} for c in callouts],
        "labels": callouts[:2],
        "image_path": source_path,
        "visual_role": candidate.visual_role,
        "source_page": candidate.page,
        "caption": candidate.nearby_caption,
    }
    screen_plan = {
        "headline": headline,
        "callouts": callouts,
        "labels": callouts[:2],
        "highlight_regions": [],
        "max_screen_text": 80,
        "layout_hint": kind,
    }
    narration = _visual_narration(candidate, topic, kind)
    return ScriptScene(
        scene_id=f"ch_vis_sc_{scene_no:03d}",
        chapter_id="ch_vis",
        title=topic[:40],
        source_pages=[candidate.page],
        source_refs=[f"p.{candidate.page}"],
        narration_text_zh_tw=narration,
        on_screen_text="\n".join([headline] + callouts[:2]),
        visual_hint=f"source visual walkthrough: {candidate.visual_role}",
        visual_type=VisualType.sketchbook_card,
        scene_kind=kind,
        facts=[],
        evidence_spans=_source_evidence(candidate),
        layout_payload=payload,
        visual_anchor=anchor,
        screen_plan=screen_plan,
        importance="high" if candidate.salience_score >= 4.0 else "medium",
        estimated_duration_sec=_duration_for_kind(kind),
    )


def _should_build_comparison(left: VisualCandidate,
                             right: VisualCandidate) -> bool:
    if right.page - left.page > 2:
        return False
    blob = " ".join([
        left.nearby_heading or "", left.nearby_caption or "", left.nearby_text or "",
        right.nearby_heading or "", right.nearby_caption or "", right.nearby_text or "",
    ])
    return any(term in blob for term in ("比較", "對比", "before", "after", "淺", "深", "高", "低", "冷", "暖"))


def _build_comparison_visual_scene(*, left: VisualCandidate,
                                   right: VisualCandidate,
                                   sources: ChunkedSources,
                                   scene_no: int) -> ScriptScene:
    page_by_no = {p.page: p for p in sources.pages}
    left_topic = _visual_topic_title(left, page_by_no)
    right_topic = _visual_topic_title(right, page_by_no)
    title = _short_headline(left_topic if left_topic == right_topic else f"{left_topic} 比較")
    left_label = _comparison_label(left, fallback="左邊")
    right_label = _comparison_label(right, fallback="右邊")
    payload = {
        "headline": title,
        "visuals": [
            {"image_path": candidate_source_path(left), "label": left_label, "page": left.page},
            {"image_path": candidate_source_path(right), "label": right_label, "page": right.page},
        ],
        "left_label": left_label,
        "right_label": right_label,
    }
    narration = (
        f"這裡用左右兩個來源例子來看「{left_topic}」。"
        f"左邊先看{left_label}，右邊再看{right_label}。"
        "真正要注意的不是文字標籤，而是兩張圖在效果、設定或結果上的差異。"
    )
    evidence = _source_evidence(left) + _source_evidence(right)
    return ScriptScene(
        scene_id=f"ch_vis_sc_{scene_no:03d}",
        chapter_id="ch_vis",
        title=title[:40],
        source_pages=sorted({left.page, right.page}),
        source_refs=[f"p.{p}" for p in sorted({left.page, right.page})],
        narration_text_zh_tw=narration,
        on_screen_text=f"{title}\n{left_label} / {right_label}",
        visual_hint="source comparison walkthrough",
        visual_type=VisualType.sketchbook_card,
        scene_kind="comparison_visual_card",
        facts=[],
        evidence_spans=evidence,
        layout_payload=payload,
        visual_anchor=anchor_from_candidate(left, why="comparison left source visual"),
        screen_plan={
            "headline": title,
            "labels": [left_label, right_label],
            "callouts": ["看差異", "連回設定"],
            "highlight_regions": [],
            "max_screen_text": 70,
            "layout_hint": "comparison_visual_card",
        },
        importance="high",
        estimated_duration_sec=_duration_for_kind("comparison_visual_card"),
    )


def _comparison_label(candidate: VisualCandidate, *, fallback: str) -> str:
    text = " ".join([
        candidate.nearby_heading or "",
        candidate.nearby_caption or "",
        candidate.nearby_text or "",
    ])
    labels = (
        ("淺", "淺景深"), ("深", "深景深"), ("高", "高設定"),
        ("低", "低設定"), ("冷", "偏冷"), ("暖", "偏暖"),
        ("before", "Before"), ("after", "After"),
    )
    lower = text.lower()
    for needle, label in labels:
        if needle.lower() in lower:
            return label
    return fallback


def _short_headline(text: str) -> str:
    text = str(text or "").strip().replace("\n", " ")
    return text[:18] if len(text) > 18 else text


def _visual_narration(candidate: VisualCandidate, topic: str, kind: str) -> str:
    role = candidate.visual_role or "source visual"
    heading = candidate.nearby_heading or topic
    if kind == "source_table_explainer":
        return (
            f"這裡先看第 {candidate.page} 頁的表格或資料區。"
            f"它和「{heading}」有關，重點是先找欄位代表什麼，"
            "再看哪一列會影響你的判斷。不要急著把整張表背起來。"
        )
    if kind == "source_screenshot_explainer":
        return (
            f"畫面上這張來源截圖對應「{heading}」。"
            "你可以先找按鈕、選單或指標的位置，再理解這個設定為什麼會改變結果。"
        )
    if kind == "process_visual_card":
        return (
            f"這張圖想告訴你「{heading}」的操作順序。"
            "先看第一步要調整什麼，再看後面的結果如何連動；新手最容易漏掉的是步驟之間的因果。"
        )
    if kind == "figure_explainer":
        return (
            f"這裡真正要注意的是第 {candidate.page} 頁這張圖的關係。"
            f"它不是裝飾圖，而是在說明「{heading}」。"
            "請把視線放在形狀、方向或差異上，再回頭理解文字定義。"
        )
    return (
        f"畫面中這個來源例子來自第 {candidate.page} 頁，主題是「{heading}」。"
        "先觀察圖中的主體和差異，再聽旁白補上原因；這樣會比只讀文字卡更容易記住。"
    )


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
    if profile.document_visual_rich:
        narration = (
            f"歡迎收看這份{eyebrow}。我們會用大約"
            f"{max(1, round(outline.target_minutes))}分鐘，"
            "直接看來源裡的圖片、圖表、截圖或範例，"
            "一段一段理解它們想教你的重點。"
        )
    elif profile.doc_kind in {DocKind.contract, DocKind.form, DocKind.policy}:
        narration = (
            f"歡迎收看這份{eyebrow}。我們會用大約"
            f"{max(1, round(outline.target_minutes))}分鐘，"
            "把這份文件最關鍵的時程、條款與注意事項，"
            "整理成幾張清楚的卡片。"
        )
    else:
        narration = (
            f"歡迎收看這份{eyebrow}。我們會用大約"
            f"{max(1, round(outline.target_minutes))}分鐘，"
            "把來源文件的主題、例子與重點整理成清楚的導讀。"
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
    inserted = 0
    out: list[ScriptScene] = []
    for sc in scenes:
        out.append(sc)
        n = counts.get(sc.scene_id, 0)
        for _ in range(n):
            emitted[sc.scene_id] = emitted.get(sc.scene_id, 0) + 1
            expansion = _build_expansion_scene(sc, emitted[sc.scene_id])
            if expansion is None:
                continue
            out.append(expansion)
            inserted += 1
    if inserted < len(pad_ids):
        plan_report["under_target_reason"] = (
            "not enough distinct grounded facts/evidence to expand without "
            "repeating content"
        )
        plan_report["dropped_weak_expansion_count"] = len(pad_ids) - inserted
    plan_report["inserted_expansion_scene_count"] = inserted
    return out


def _build_expansion_scene(base: ScriptScene, seq: int) -> ScriptScene | None:
    kind = (base.scene_kind or "").lower()
    if kind == "penalty_table":
        rows = list((base.layout_payload or {}).get("rows") or [])
        item = _nth_grounded_item(rows, seq)
        if item is None:
            return None
        return _item_expansion_scene(
            base, item,
            scene_kind="penalty_table",
            title_prefix="罰則細節",
            label="罰則",
            value=str(item.get("value") or ""),
            body=_penalty_item_body(item),
            layout_payload={"rows": [item]},
            narration=_penalty_item_narration(item),
            duration=18.0,
        )
    if kind == "risk_warning":
        items = list((base.layout_payload or {}).get("items") or [])
        item = _nth_grounded_item(items, seq)
        if item is None:
            return None
        return _item_expansion_scene(
            base, item,
            scene_kind="risk_warning",
            title_prefix="風險細節",
            label="風險",
            value=_short_item_value(item),
            body=str(item.get("text") or ""),
            layout_payload={"items": [item]},
            narration=_risk_item_narration(item),
            duration=18.0,
        )
    if kind == "checklist":
        items = list((base.layout_payload or {}).get("items") or [])
        item = _nth_grounded_item(items, seq)
        if item is None:
            return None
        group = _checklist_group(item)
        return _item_expansion_scene(
            base, item,
            scene_kind="checklist",
            title_prefix=f"應辦事項：{group}",
            label="應辦事項",
            value=_short_item_value(item),
            body=str(item.get("text") or ""),
            layout_payload={"items": [item], "group": group},
            narration=_checklist_item_narration(item, group),
            duration=18.0,
        )
    if kind == "deadline_timeline":
        events = list((base.layout_payload or {}).get("events") or [])
        item = _nth_grounded_item(events, seq)
        if item is None:
            return None
        return _item_expansion_scene(
            base, item,
            scene_kind="deadline_timeline",
            title_prefix="時程細節",
            label="期限",
            value=str(item.get("value") or ""),
            body=_deadline_item_body(item),
            layout_payload={"events": [item]},
            narration=_deadline_item_narration(item),
            duration=18.0,
        )
    if kind == "key_number":
        items = list((base.layout_payload or {}).get("items") or [])
        item = _nth_grounded_item(items, seq)
        if item is None:
            return None
        return _item_expansion_scene(
            base, item,
            scene_kind="key_number",
            title_prefix="關鍵數字",
            label=str(item.get("label") or "關鍵數字"),
            value=str(item.get("value") or ""),
            body=str(item.get("context") or item.get("value") or ""),
            layout_payload={"items": [item]},
            narration=_keynumber_narration([item]),
            duration=16.0,
        )
    if kind == "do_dont":
        do_items = [{**it, "_column": "do"} for it in (base.layout_payload or {}).get("do") or []]
        dont_items = [{**it, "_column": "dont"} for it in (base.layout_payload or {}).get("dont") or []]
        item = _nth_grounded_item(do_items + dont_items, seq)
        if item is None:
            return None
        clean = {k: v for k, v in item.items() if k != "_column"}
        payload = {"do": [clean], "dont": []} if item.get("_column") == "do" else {"do": [], "dont": [clean]}
        label = "應做" if item.get("_column") == "do" else "不要做"
        return _item_expansion_scene(
            base, clean,
            scene_kind="do_dont",
            title_prefix=label,
            label=label,
            value=_short_item_value(clean),
            body=str(clean.get("text") or ""),
            layout_payload=payload,
            narration=f"這一點請特別記住：{str(clean.get('text') or '')[:80]}。",
            duration=18.0,
        )
    return None


def _nth_grounded_item(items: list[dict], seq: int) -> dict | None:
    grounded: list[dict] = []
    seen: set[tuple[int, str, str, str]] = set()
    for it in items:
        if not isinstance(it, dict) or not isinstance(it.get("page"), int):
            continue
        key = (
            it["page"],
            _item_quote(it),
            str(it.get("condition") or ""),
            str(it.get("value") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        grounded.append(it)
    if 1 <= seq <= len(grounded):
        return dict(grounded[seq - 1])
    return None


def _item_expansion_scene(base: ScriptScene, item: dict, *, scene_kind: str,
                          title_prefix: str, label: str, value: str,
                          body: str, layout_payload: dict,
                          narration: str, duration: float) -> ScriptScene | None:
    page = item.get("page")
    quote = _item_quote(item)
    if not isinstance(page, int) or not quote:
        return None
    evidence = [EvidenceSpan(
        page=page,
        quote=quote[:160],
        label=label,
        value=value or None,
        importance="high" if (base.importance or "") == "high" else "medium",
    )]
    facts = []
    if value:
        facts.append(Fact(
            label=label,
            value=value[:80],
            importance=base.importance or "medium",
            evidence_index=0,
        ))
    focus = _item_focus(item, value=value, fallback=title_prefix)
    payload = dict(layout_payload)
    payload.update({
        "expansion_of": base.scene_id,
        "source_kind": base.scene_kind,
        "body": body[:360],
    })
    suffix = hashlib.sha1(
        f"{base.scene_id}|{page}|{focus}|{quote}".encode("utf-8")
    ).hexdigest()[:8]
    return ScriptScene(
        scene_id=f"{base.scene_id}_ex_{suffix}",
        chapter_id=base.chapter_id,
        title=f"{title_prefix} · {focus}"[:40],
        source_pages=[page],
        source_refs=[f"p.{page}"],
        narration_text_zh_tw=narration[:600],
        on_screen_text=focus[:30],
        visual_hint=f"grounded {scene_kind} expansion",
        visual_type=VisualType.sketchbook_card,
        scene_kind=scene_kind,
        facts=facts,
        evidence_spans=evidence,
        layout_payload=payload,
        importance=base.importance or "medium",
        estimated_duration_sec=duration,
    )


def _item_quote(item: dict) -> str:
    return str(
        item.get("text") or item.get("context") or item.get("condition") or
        item.get("label") or item.get("value") or ""
    ).strip()


def _item_focus(item: dict, *, value: str, fallback: str) -> str:
    condition = str(item.get("condition") or "").strip()
    text = str(item.get("text") or item.get("label") or "").strip()
    if condition and value:
        return f"{condition} → {value}"[:36]
    if value:
        return value[:36]
    return (condition or text or fallback)[:36]


def _short_item_value(item: dict) -> str:
    raw = str(item.get("value") or item.get("text") or item.get("condition") or "").strip()
    return raw[:48]


def _penalty_item_body(item: dict) -> str:
    condition = str(item.get("condition") or "")
    value = str(item.get("value") or "")
    return f"{condition}\n費用 / 比例：{value}".strip()


def _penalty_item_narration(item: dict) -> str:
    condition = str(item.get("condition") or "這個情境")
    value = str(item.get("value") or "文件列出的費用")
    return f"罰則細節：{condition}，文件列出的費用或比例是 {value}。請確認自己的取消或變更時間是否落在這一列。"


def _risk_item_narration(item: dict) -> str:
    text = str(item.get("text") or "這項風險")
    return f"這是一項需要單獨記住的風險：{text[:100]}。處理前請回到來源頁確認完整條件。"


def _checklist_group(item: dict) -> str:
    text = str(item.get("text") or "")
    groups = (
        ("付款", ("付款", "繳付", "繳納", "訂金", "尾款", "全額費")),
        ("名單 / 艙房", ("名單", "分房", "艙房", "更改", "更動", "改名")),
        ("護照 / 簽證", ("護照", "簽證", "登船", "入境")),
        ("保險 / 健康", ("保險", "疾病", "醫師", "適航", "投保", "服藥")),
    )
    for label, needles in groups:
        if any(n in text for n in needles):
            return label
    return "確認事項"


def _checklist_item_narration(item: dict, group: str) -> str:
    text = str(item.get("text") or "這項應辦事項")
    return f"應辦事項，{group}：{text[:100]}。這張卡只列這一件事，方便你逐項核對。"


def _deadline_item_body(item: dict) -> str:
    value = str(item.get("value") or "")
    label = str(item.get("label") or "")
    return f"{value}\n{label}".strip()


def _deadline_item_narration(item: dict) -> str:
    value = str(item.get("value") or "這個期限")
    label = str(item.get("label") or "文件中的期限條件")
    return f"時程細節：{value}。它對應的來源脈絡是：{label[:100]}。請避免只記日期而忽略條件。"


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
        "source_visual_explainer": 26.0,
        "figure_explainer": 28.0,
        "comparison_visual_card": 28.0,
        "process_visual_card": 28.0,
        "source_table_explainer": 28.0,
        "source_screenshot_explainer": 28.0,
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
