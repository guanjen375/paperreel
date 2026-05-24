"""Heuristic fact extractor.

The sketchbook pipeline must never invent numbers / dates / fees /
percentages. This module pulls candidate facts out of the *actual* page
text and tags each with the originating page so the validator can
verify the quote against the ingested PDF.

It is intentionally regex-based and conservative — false positives are
OK because the LLM can decline to use a candidate, but a fabricated
fact has to be impossible to introduce here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import EvidenceSpan, Fact, Importance


_CJK_NUM = r"[零〇一二兩三四五六七八九十百千萬]+"


@dataclass(frozen=True)
class _Pattern:
    kind: str
    regex: re.Pattern[str]
    label: str
    importance: Importance


# Order matters when patterns overlap (e.g. a deadline that also
# contains a percentage). We try longest / most specific first.
_PATTERNS: tuple[_Pattern, ...] = (
    _Pattern("deadline", re.compile(
        r"(?:於|在|限|應於)?\s*(\d+\s*(?:天|日|個月|月|年|工作天|工作日))\s*(?:內|前|以內)",
    ), "期限", Importance.high),
    _Pattern("deadline", re.compile(
        r"(?:within|no later than)\s+(\d+\s+(?:day|days|week|weeks|month|months|year|years))",
        re.IGNORECASE,
    ), "Deadline", Importance.high),
    _Pattern("deadline_cjk", re.compile(
        r"(" + _CJK_NUM + r"(?:天|日|個月|月|年))(?:內|前|以內)",
    ), "期限", Importance.high),
    _Pattern("absolute_date", re.compile(
        r"(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)",
    ), "日期", Importance.medium),
    _Pattern("absolute_date", re.compile(
        r"(\d{4}-\d{2}-\d{2})",
    ), "日期", Importance.medium),
    _Pattern("percent", re.compile(
        r"(\d{1,3}(?:\.\d+)?\s*%)",
    ), "百分比", Importance.high),
    _Pattern("percent_cjk", re.compile(
        r"(百分之[一二三四五六七八九十百零〇\d]+)",
    ), "百分比", Importance.high),
    _Pattern("money", re.compile(
        r"((?:NT\$|US\$|USD|TWD|新台幣|台幣|美金|美元|歐元|港幣|人民幣)\s*[\d,]+(?:\.\d+)?(?:\s*(?:元|圓))?)",
    ), "金額", Importance.high),
    _Pattern("money_plain", re.compile(
        r"(\d{1,3}(?:,\d{3})+\s*(?:元|圓|塊))",
    ), "金額", Importance.high),
    _Pattern("count", re.compile(
        r"(\d+\s*(?:人|件|份|名|張|筆|次|期|頁))",
    ), "數量", Importance.medium),
)


_BAD_TAIL_CHARS = "，。、；：,.;:!?！？\n\r\t"


def _trim_quote(s: str, max_len: int = 80) -> str:
    s = s.strip().strip(_BAD_TAIL_CHARS).strip()
    if len(s) <= max_len:
        return s
    return s[:max_len].rstrip(_BAD_TAIL_CHARS)


def _surrounding_quote(page_text: str, match: re.Match[str], *,
                       window: int = 60) -> str:
    """Pull a quote that brackets the matched value with surrounding
    context, so the renderer can show a meaningful snippet (not just
    the bare number). Capped at 80 chars after trimming punctuation."""
    start = max(0, match.start() - window)
    end = min(len(page_text), match.end() + window)
    snippet = page_text[start:end]
    # Try to break at the nearest sentence-ish boundary on each side.
    left = snippet[: match.start() - start]
    right = snippet[match.end() - start:]
    for sep in ("。", "．", ".", "！", "!", "？", "?", "；", ";"):
        if sep in left:
            left = left[left.rindex(sep) + 1:]
        if sep in right:
            right = right[: right.index(sep) + 1]
    return _trim_quote(left + match.group(0) + right)


def extract_from_page(text: str, page: int) -> list[tuple[Fact, EvidenceSpan]]:
    """Return ``(Fact, EvidenceSpan)`` pairs found in ``text``.

    Each fact's value is the raw matched substring; the evidence span
    quotes a window of context around it. The pairing is kept so the
    caller can preserve `Fact.evidence_index` when adding both into a
    scene.
    """
    out: list[tuple[Fact, EvidenceSpan]] = []
    seen_spans: set[tuple[int, int]] = set()
    for pat in _PATTERNS:
        for m in pat.regex.finditer(text):
            key = (m.start(), m.end())
            if key in seen_spans:
                continue
            seen_spans.add(key)
            value = _trim_quote(m.group(1) if m.groups() else m.group(0), max_len=40)
            if not value:
                continue
            quote = _surrounding_quote(text, m)
            if not quote:
                continue
            fact = Fact(
                label=pat.label,
                value=value,
                importance=pat.importance.value,
            )
            span = EvidenceSpan(
                page=page,
                quote=quote,
                label=pat.label,
                value=value,
                importance=pat.importance.value,
            )
            out.append((fact, span))
    return out


def extract_from_pages(pages: dict[int, str]) -> dict[int, list[tuple[Fact, EvidenceSpan]]]:
    """Apply ``extract_from_page`` to a {page_no: text} mapping."""
    return {p: extract_from_page(text, p) for p, text in pages.items()}


# ---------- scene-kind specific filters ----------

_TIMELINE_KINDS = {"deadline", "deadline_cjk", "absolute_date"}
_PENALTY_KEYWORDS = ("違約", "解約", "賠償", "退款", "罰款", "penalty", "refund")
_RISK_KEYWORDS = ("風險", "警告", "注意", "禁止", "不得", "warning", "caution")
_OBLIGATION_KEYWORDS = ("應", "需", "必須", "提供", "繳交", "shall", "must")


def group_for_scene_kind(scene_kind: str,
                         page_facts: dict[int, list[tuple[Fact, EvidenceSpan]]],
                         page_text: dict[int, str],
                         *, max_items: int = 6) -> dict:
    """Pick the right facts for a given sketchbook scene_kind.

    Returns a ``layout_payload``-shaped dict the renderer understands.
    The exact schema per kind is documented in the renderer.
    """
    if scene_kind == "deadline_timeline":
        events: list[dict] = []
        for page, items in sorted(page_facts.items()):
            for fact, span in items:
                if fact.label in ("期限", "Deadline", "日期"):
                    events.append({
                        "value": fact.value,
                        "label": _short_context(span.quote, fact.value),
                        "page": page,
                    })
        return {"events": events[:max_items]}

    if scene_kind == "penalty_table":
        rows: list[dict] = []
        for page, items in sorted(page_facts.items()):
            text = page_text.get(page, "")
            for fact, span in items:
                if fact.label in ("百分比", "金額") and any(
                    kw in span.quote for kw in _PENALTY_KEYWORDS
                ) or _line_has_penalty(text, fact.value):
                    rows.append({
                        "condition": _short_context(span.quote, fact.value),
                        "value": fact.value,
                        "page": page,
                    })
        return {"rows": rows[:max_items]}

    if scene_kind == "checklist":
        items_out: list[dict] = []
        for page, items in sorted(page_facts.items()):
            text = page_text.get(page, "")
            for fact, span in items:
                if any(kw in span.quote for kw in _OBLIGATION_KEYWORDS):
                    items_out.append({
                        "text": _short_context(span.quote, fact.value),
                        "value": fact.value,
                        "page": page,
                    })
        # Fall back: pull obligation sentences with no number attached.
        if len(items_out) < 3:
            for page, text in page_text.items():
                for sent in _split_sentences(text):
                    if any(kw in sent for kw in _OBLIGATION_KEYWORDS):
                        items_out.append({
                            "text": sent.strip()[:60],
                            "value": "",
                            "page": page,
                        })
                        if len(items_out) >= max_items:
                            break
                if len(items_out) >= max_items:
                    break
        return {"items": items_out[:max_items]}

    if scene_kind == "risk_warning":
        risks: list[dict] = []
        for page, text in page_text.items():
            for sent in _split_sentences(text):
                if any(kw in sent for kw in _RISK_KEYWORDS):
                    risks.append({
                        "text": sent.strip()[:80],
                        "page": page,
                    })
        return {"items": risks[:max_items]}

    if scene_kind == "do_dont":
        do_items: list[dict] = []
        dont_items: list[dict] = []
        for page, text in page_text.items():
            for sent in _split_sentences(text):
                if any(kw in sent for kw in ("不得", "禁止", "勿", "請勿")):
                    dont_items.append({"text": sent.strip()[:60], "page": page})
                elif any(kw in sent for kw in ("應", "需", "必須", "請")):
                    do_items.append({"text": sent.strip()[:60], "page": page})
        return {
            "do": do_items[:max_items],
            "dont": dont_items[:max_items],
        }

    if scene_kind == "key_number":
        nums: list[dict] = []
        for page, items in sorted(page_facts.items()):
            for fact, span in items:
                if fact.label in ("百分比", "金額", "數量"):
                    nums.append({
                        "label": fact.label,
                        "value": fact.value,
                        "context": _short_context(span.quote, fact.value),
                        "page": page,
                    })
        return {"items": nums[:max_items]}

    return {}


def _short_context(quote: str, anchor: str, *, max_len: int = 60) -> str:
    """Return a shortened quote that still surrounds ``anchor`` if possible."""
    if anchor and anchor in quote:
        idx = quote.index(anchor)
        start = max(0, idx - 20)
        end = min(len(quote), idx + len(anchor) + 30)
        return _trim_quote(quote[start:end], max_len=max_len)
    return _trim_quote(quote, max_len=max_len)


_SENT_SPLIT = re.compile(r"(?<=[。．.!?！？；;])\s*")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text) if s.strip()]


def _line_has_penalty(text: str, value: str) -> bool:
    if not value or value not in text:
        return False
    idx = text.index(value)
    window = text[max(0, idx - 40): idx + len(value) + 40]
    return any(kw in window for kw in _PENALTY_KEYWORDS)
