"""Heuristic document classifier.

Looks at extracted page text + structural signals (page count, average
density, heading density, image density) and emits a :class:`DocProfile`.
LLM refinement is optional and lives in :func:`classify_with_llm`.

This module has no LLM / network dependency: the basic classifier must
work on a pure-CPU machine so the sketchbook pipeline boots without
Ollama running.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ..models import ChunkedSources, DocKind, DocProfile


# Keywords are intentionally short and broad. We weight by category to
# avoid one stray word in a contract pretending to be a paper. The
# heuristic is biased toward Traditional/Simplified Chinese plus the
# English markers that appear in academic / technical docs.
_KEYWORDS: dict[DocKind, list[str]] = {
    DocKind.contract: [
        "甲方", "乙方", "丙方", "雙方", "簽署", "簽訂", "簽約",
        "本合約", "本契約", "本協議", "條款", "違約", "違約金",
        "賠償", "解約", "終止", "授權", "授權範圍", "費用",
        "付款", "預付", "尾款", "保證金", "押金", "訂金",
        "取消費用", "取消者", "退費", "不予退款", "概不負責",
        "旅客", "郵輪", "護照", "簽證",
        "agreement", "contract", "shall", "hereby", "obligation",
        "warranty", "indemnif",
    ],
    DocKind.form: [
        "申請書", "申請表", "請填寫", "請勾選", "個人資料",
        "姓名", "身分證", "護照", "簽名", "蓋章", "日期",
        "聯絡電話", "電子郵件", "地址", "□", "☐",
        "form", "please fill", "tick",
    ],
    DocKind.paper: [
        "abstract", "introduction", "method", "methods", "methodology",
        "experiment", "experiments", "result", "results", "discussion",
        "conclusion", "references", "related work", "ablation",
        "摘要", "緒論", "方法", "實驗", "結果", "結論", "參考文獻",
    ],
    DocKind.manual: [
        "步驟", "安裝", "設定", "啟動", "操作", "故障", "排除",
        "注意事項", "注意", "警告", "前置作業", "前置條件",
        "操作說明", "使用說明", "保固",
        "install", "setup", "step", "guide", "troubleshoot",
        "warning", "caution",
    ],
    DocKind.report: [
        "摘要", "本季", "本年", "年度", "指標", "趨勢", "成長",
        "分析", "建議", "風險", "績效", "報告", "市場",
        "executive summary", "kpi", "quarter", "annual",
        "forecast", "performance", "trend",
    ],
    DocKind.policy: [
        "政策", "辦法", "規範", "條例", "守則", "適用範圍",
        "適用對象", "本辦法", "本規範", "本守則", "違反",
        "policy", "compliance", "applies to", "regulation",
        "防疫措施", "公告", "個人資料保護",
    ],
    DocKind.slides: [
        # Slides rarely carry the words above; the structural detector
        # handles the bulk of the signal. Empty list keeps the loop
        # working without false positives.
    ],
}


# Patterns that strongly hint at scheduled obligations (dates,
# percentages, currency). Useful to nudge "report" vs "contract".
_DEADLINE_PATTERNS = [
    re.compile(r"\d+\s*(天|日)內"),
    re.compile(r"\d+\s*(個月|月)內"),
    re.compile(r"\d+\s*(個|個)?(工作天|工作日)"),
    re.compile(r"於\s*\d+\s*(天|日|月)"),
    re.compile(r"(within|no later than)\s+\d+\s+(day|days|week|weeks|month|months)",
               re.IGNORECASE),
]
_PERCENT_PATTERN = re.compile(r"\d{1,3}\s*%|百分之[一二三四五六七八九十百零〇\d]+")
_MONEY_PATTERN = re.compile(
    r"(NT\$|US\$|USD|TWD|新台幣|台幣|美元|歐元|港幣)\s*[\d,]+|"
    r"\d{1,3}(,\d{3})+\s*(元|圓|塊)",
)


# Storyboard skeletons per doc_kind. Each entry is a scene_kind tag the
# script writer + renderer understand. Cover/recap always book-end.
_VISUAL_STORYBOARD = [
    "cover", "section_intro", "source_visual_explainer",
    "figure_explainer", "process_visual_card", "comparison_visual_card",
    "source_table_explainer", "source_screenshot_explainer",
    "checklist", "recap_card",
]


_VISUAL_TUTORIAL_TERMS = (
    "步驟", "技巧", "設定", "模式", "範例", "實戰", "如何",
    "操作", "練習", "示意", "對比", "比較", "畫面", "截圖",
    "光圈", "快門", "iso", "感光度", "曝光", "景深", "直方圖",
    "白平衡", "對焦", "測光", "鏡頭", "camera", "photo",
    "tutorial", "workflow", "example", "screenshot",
)


_STORYBOARDS: dict[DocKind, list[str]] = {
    DocKind.contract: [
        "cover", "section_intro", "deadline_timeline", "penalty_table",
        "checklist", "risk_warning", "do_dont", "key_number", "recap_card",
    ],
    DocKind.form: [
        "cover", "section_intro", "checklist", "deadline_timeline",
        "do_dont", "risk_warning", "recap_card",
    ],
    DocKind.paper: [
        "cover", "section_intro", "paragraph_card", "paragraph_card",
        "paragraph_card", "checklist", "recap_card",
    ],
    DocKind.manual: [
        "cover", "section_intro", "checklist", "risk_warning",
        "do_dont", "paragraph_card", "recap_card",
    ],
    DocKind.report: [
        "cover", "section_intro", "key_number", "key_number",
        "risk_warning", "checklist", "recap_card",
    ],
    DocKind.policy: [
        "cover", "section_intro", "checklist", "risk_warning",
        "deadline_timeline", "do_dont", "recap_card",
    ],
    DocKind.slides: [
        "cover", "section_intro", "paragraph_card", "paragraph_card",
        "checklist", "recap_card",
    ],
    DocKind.unknown: [
        "cover", "section_intro", "paragraph_card", "checklist",
        "recap_card",
    ],
}


@dataclass
class _Signals:
    keyword_hits: dict[DocKind, int]
    avg_cjk_per_page: float
    headings_per_page: float
    images_per_page: float
    deadline_hits: int
    percent_hits: int
    money_hits: int
    page_count: int
    visual_tutorial_hits: int
    useful_visuals: int
    decorative_visuals: int
    large_visual_pages: int
    source_visuals_available: int
    visual_rich_score: float


def _collect_signals(sources: ChunkedSources) -> _Signals:
    text = " ".join(p.text for p in sources.pages)
    lower = text.lower()
    keyword_hits: dict[DocKind, int] = {}
    for kind, words in _KEYWORDS.items():
        keyword_hits[kind] = sum(lower.count(w.lower()) for w in words)
    deadline_hits = sum(len(p.findall(text)) for p in _DEADLINE_PATTERNS)
    percent_hits = len(_PERCENT_PATTERN.findall(text))
    money_hits = len(_MONEY_PATTERN.findall(text))
    page_count = max(1, sources.page_count)
    avg_cjk = sources.cjk_char_count / page_count
    headings_per_page = sources.heading_count / page_count
    image_count = len(sources.images) or int(getattr(sources, "image_count", 0) or 0)
    images_per_page = image_count / page_count
    tutorial_hits = sum(lower.count(t.lower()) for t in _VISUAL_TUTORIAL_TERMS)
    inventory = list(getattr(sources, "visual_inventory", []) or [])
    useful_visuals = sum(
        1 for c in inventory
        if getattr(c, "likely_useful", False)
        and getattr(c, "visual_role", "unknown") not in {"decorative", "logo", "seal", "signature"}
    )
    decorative_visuals = sum(
        1 for c in inventory
        if getattr(c, "is_decorative", False)
        or getattr(c, "visual_role", "unknown") in {"decorative", "logo", "seal", "signature"}
    )
    large_visual_pages = len({
        int(getattr(c, "page", 0) or 0) for c in inventory
        if getattr(c, "likely_useful", False)
        and float(getattr(c, "page_area_ratio", 0.0) or 0.0) >= 0.12
    })
    # Old chunked_sources.json files have no visual_inventory. Use the
    # embedded image count as a weaker fallback so classification still
    # improves after only a model load.
    if not inventory and image_count:
        useful_visuals = image_count
        large_visual_pages = min(page_count, image_count)
    source_visuals_available = useful_visuals
    visual_rich_score = _visual_rich_score(
        useful_visuals=useful_visuals,
        decorative_visuals=decorative_visuals,
        large_visual_pages=large_visual_pages,
        page_count=page_count,
        images_per_page=images_per_page,
        avg_cjk_per_page=avg_cjk,
        tutorial_hits=tutorial_hits,
        headings_per_page=headings_per_page,
    )
    return _Signals(
        keyword_hits=keyword_hits,
        avg_cjk_per_page=avg_cjk,
        headings_per_page=headings_per_page,
        images_per_page=images_per_page,
        deadline_hits=deadline_hits,
        percent_hits=percent_hits,
        money_hits=money_hits,
        page_count=page_count,
        visual_tutorial_hits=tutorial_hits,
        useful_visuals=useful_visuals,
        decorative_visuals=decorative_visuals,
        large_visual_pages=large_visual_pages,
        source_visuals_available=source_visuals_available,
        visual_rich_score=visual_rich_score,
    )


def _visual_rich_score(*, useful_visuals: int, decorative_visuals: int,
                       large_visual_pages: int, page_count: int,
                       images_per_page: float, avg_cjk_per_page: float,
                       tutorial_hits: int, headings_per_page: float) -> float:
    score = 0.0
    useful_per_page = useful_visuals / max(1, page_count)
    large_page_ratio = large_visual_pages / max(1, page_count)
    if useful_visuals >= 3:
        score += 1.0
    if useful_per_page >= 0.20:
        score += 1.5
    if useful_per_page >= 0.45:
        score += 1.0
    if large_page_ratio >= 0.18:
        score += 1.5
    if large_page_ratio >= 0.40:
        score += 1.0
    if images_per_page >= 0.45:
        score += 0.8
    if avg_cjk_per_page < 550 and (useful_per_page >= 0.15 or headings_per_page >= 0.6):
        score += 0.8
    if tutorial_hits >= 4:
        score += 0.8
    if tutorial_hits >= 10:
        score += 0.8
    if decorative_visuals > useful_visuals and useful_visuals < 3:
        score -= 1.5
    return round(max(0.0, score), 3)


def _slides_score(sig: _Signals) -> float:
    """Slides have sparse text per page + many heading-like lines.

    A two-page narrative paragraph also has "sparse text" if measured
    naively, so we require *structural* markers (headings or images)
    in addition to low density.
    """
    if sig.headings_per_page < 0.5 and sig.images_per_page < 0.3:
        return 0.0
    score = 0.0
    if sig.avg_cjk_per_page < 250:
        score += 2.0
    if sig.headings_per_page >= 0.7:
        score += 1.5
    if sig.images_per_page >= 0.4:
        score += 1.0
    return score


def _contract_bonus(sig: _Signals) -> float:
    """Contracts almost always pack money / deadlines / percentages."""
    score = 0.0
    if sig.deadline_hits >= 3:
        score += 1.5
    if sig.money_hits >= 2:
        score += 1.0
    if sig.percent_hits >= 3:
        score += 0.8
    return score


def classify(sources: ChunkedSources) -> DocProfile:
    """Pure-heuristic classification — no LLM call.

    Returns a :class:`DocProfile`. ``confidence`` is the relative gap
    between the top score and the runner-up, clamped to [0, 1].
    """
    sig = _collect_signals(sources)
    scores: Counter[DocKind] = Counter()
    for kind, hits in sig.keyword_hits.items():
        # Hits scale roughly with document length; normalise by sqrt
        # so a 600-page report doesn't always crush a 40-page contract.
        scores[kind] = hits / max(1.0, sig.page_count ** 0.5)
    scores[DocKind.slides] = _slides_score(sig)
    scores[DocKind.contract] += _contract_bonus(sig)
    visual_family_blocked = (
        scores[DocKind.contract] >= 2.0
        or scores[DocKind.form] >= 2.0
        or scores[DocKind.policy] >= 2.0
    )
    if not visual_family_blocked:
        scores[DocKind.manual] += min(3.0, sig.visual_tutorial_hits / max(2.0, sig.page_count ** 0.5) * 0.8)
        if sig.visual_rich_score >= 3.0 and sig.visual_tutorial_hits >= 2:
            scores[DocKind.manual] += 1.5
        if sig.visual_rich_score >= 3.0 and sig.avg_cjk_per_page < 300:
            scores[DocKind.slides] += 1.0
    # Reports lean on percent + money but in a non-obligation way; if
    # contract keywords aren't present, treat heavy numbers as report.
    if scores[DocKind.contract] < 1.0 and (sig.percent_hits + sig.money_hits) >= 5:
        scores[DocKind.report] += 1.5
    # Papers need at least one structural marker before they can win.
    if sig.keyword_hits[DocKind.paper] < 2:
        scores[DocKind.paper] *= 0.6
    ranked = scores.most_common()
    if not ranked or ranked[0][1] <= 0:
        return DocProfile(
            doc_kind=DocKind.unknown,
            confidence=0.0,
            rationale="no signals matched any document family",
            keyword_hits={k.value: v for k, v in sig.keyword_hits.items()},
            structural_hits={
                "avg_cjk_per_page": round(sig.avg_cjk_per_page, 1),
                "headings_per_page": round(sig.headings_per_page, 2),
                "images_per_page": round(sig.images_per_page, 2),
                "deadline_hits": sig.deadline_hits,
                "percent_hits": sig.percent_hits,
                "money_hits": sig.money_hits,
                "visual_tutorial_hits": sig.visual_tutorial_hits,
                "useful_visuals": sig.useful_visuals,
                "decorative_visuals": sig.decorative_visuals,
                "large_visual_pages": sig.large_visual_pages,
                "visual_rich_score": sig.visual_rich_score,
            },
            suggested_storyboard=_STORYBOARDS[DocKind.unknown],
            document_visual_rich=False,
            visual_tutorial=False,
            visual_rich_score=sig.visual_rich_score,
            source_visuals_available=sig.source_visuals_available,
        )
    top_kind, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - second_score
    # Confidence = margin / top, clamped so the LLM refinement step
    # knows how much wiggle room it has.
    confidence = 0.0 if top_score <= 0 else max(0.0, min(1.0, margin / top_score))
    rationale = (
        f"top={top_kind.value}({top_score:.2f}) "
        f"runner-up={ranked[1][0].value}({second_score:.2f}) "
        f"margin={margin:.2f}"
        if len(ranked) > 1
        else f"top={top_kind.value}({top_score:.2f})"
    )
    document_visual_rich = (
        sig.visual_rich_score >= 3.0
        and sig.source_visuals_available >= 2
        and top_kind not in {DocKind.contract, DocKind.form, DocKind.policy}
    )
    visual_tutorial = bool(
        document_visual_rich
        and (top_kind in {DocKind.manual, DocKind.slides} or sig.visual_tutorial_hits >= 2)
    )
    storyboard = (
        list(_VISUAL_STORYBOARD)
        if document_visual_rich
        else _STORYBOARDS[top_kind]
    )
    return DocProfile(
        doc_kind=top_kind,
        confidence=round(confidence, 3),
        rationale=rationale + (f" visual_rich={sig.visual_rich_score:.2f}" if document_visual_rich else ""),
        keyword_hits={k.value: v for k, v in sig.keyword_hits.items()},
        structural_hits={
            "avg_cjk_per_page": round(sig.avg_cjk_per_page, 1),
            "headings_per_page": round(sig.headings_per_page, 2),
            "images_per_page": round(sig.images_per_page, 2),
            "deadline_hits": sig.deadline_hits,
            "percent_hits": sig.percent_hits,
            "money_hits": sig.money_hits,
            "visual_tutorial_hits": sig.visual_tutorial_hits,
            "useful_visuals": sig.useful_visuals,
            "decorative_visuals": sig.decorative_visuals,
            "large_visual_pages": sig.large_visual_pages,
            "visual_rich_score": sig.visual_rich_score,
        },
        suggested_storyboard=storyboard,
        document_visual_rich=document_visual_rich,
        visual_tutorial=visual_tutorial,
        visual_rich_score=sig.visual_rich_score,
        source_visuals_available=sig.source_visuals_available,
    )


def storyboard_for(doc_kind: DocKind) -> list[str]:
    """Return the static storyboard skeleton for a given doc_kind."""
    return list(_STORYBOARDS.get(doc_kind, _STORYBOARDS[DocKind.unknown]))
