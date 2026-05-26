"""Helpers for source-visual walkthrough planning and review."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from ..models import ScreenPlan, VisualAnchor, VisualCandidate


VISUAL_FIRST_SCENE_KINDS = {
    "source_visual_explainer",
    "comparison_visual_card",
    "process_visual_card",
    "figure_explainer",
    "source_table_explainer",
    "source_screenshot_explainer",
}

SOURCE_VISUAL_ROLES = {
    "source_photo",
    "source_diagram",
    "source_table",
    "source_screenshot",
    "source_chart",
    "source_page_crop",
    "unknown",
}

_DECORATIVE_ROLES = {"decorative", "logo", "seal", "signature"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[㐀-鿿]")


def is_visual_first_kind(kind: str | None) -> bool:
    return (kind or "").lower() in VISUAL_FIRST_SCENE_KINDS


def is_source_visual_role(role: str | None) -> bool:
    return (role or "unknown") in SOURCE_VISUAL_ROLES


def candidate_source_path(candidate: VisualCandidate | dict | None) -> str | None:
    if candidate is None:
        return None
    for key in ("crop_path", "image_path", "page_render_path"):
        val = candidate.get(key) if isinstance(candidate, dict) else getattr(candidate, key, None)
        if val:
            return str(val)
    return None


def useful_candidates(candidates: list[VisualCandidate]) -> list[VisualCandidate]:
    return [
        c for c in candidates
        if c.likely_useful and c.visual_role not in _DECORATIVE_ROLES
        and candidate_source_path(c)
    ]


def anchor_from_candidate(candidate: VisualCandidate, *, why: str = "") -> VisualAnchor:
    return VisualAnchor(
        page=candidate.page,
        image_path=candidate.image_path,
        page_render_path=candidate.page_render_path,
        crop_path=candidate.crop_path,
        bbox=candidate.bbox,
        visual_role=candidate.visual_role,
        caption=candidate.nearby_caption,
        source_quote=candidate.source_quote,
        nearby_heading=candidate.nearby_heading,
        why_this_visual=why or _default_why(candidate),
    )


def anchor_source_path(anchor: VisualAnchor | dict | None) -> str | None:
    if anchor is None:
        return None
    for key in ("crop_path", "image_path", "page_render_path"):
        val = anchor.get(key) if isinstance(anchor, dict) else getattr(anchor, key, None)
        if val:
            return str(val)
    return None


def screen_plan_to_text(screen_plan: ScreenPlan | dict | None,
                        on_screen_text: str | None = None,
                        layout_payload: dict[str, Any] | None = None) -> str:
    parts: list[str] = []
    if on_screen_text:
        parts.append(on_screen_text)
    if screen_plan is not None:
        if isinstance(screen_plan, dict):
            plan = screen_plan
        else:
            plan = screen_plan.model_dump(mode="json")
        parts.append(str(plan.get("headline") or ""))
        parts.extend(str(x) for x in plan.get("callouts") or [])
        parts.extend(str(x) for x in plan.get("labels") or [])
    payload = layout_payload or {}
    for key in ("headline", "label", "left_label", "right_label"):
        if payload.get(key):
            parts.append(str(payload[key]))
    for key in ("callouts", "labels", "steps"):
        for item in payload.get(key) or []:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("label") or ""))
            else:
                parts.append(str(item))
    for item in payload.get("visuals") or []:
        if isinstance(item, dict):
            parts.append(str(item.get("label") or ""))
    return "\n".join(p for p in parts if p)


def narration_screen_overlap(screen_text: str, narration: str) -> float:
    """Rough overlap between screen words/phrases and narration.

    Returns the fraction of screen tokens that also appear in narration.
    This intentionally uses character-level CJK tokens plus short ngrams
    so copied paragraphs score high while short labels do not dominate.
    """
    screen_tokens = _overlap_tokens(screen_text)
    if not screen_tokens:
        return 0.0
    narration_tokens = _overlap_tokens(narration)
    if not narration_tokens:
        return 0.0
    return len(screen_tokens & narration_tokens) / max(1, len(screen_tokens))


def _overlap_tokens(text: str) -> set[str]:
    norm = unicodedata.normalize("NFKC", text or "").lower()
    tokens = _TOKEN_RE.findall(norm)
    out: set[str] = set()
    ascii_buf: list[str] = []
    cjk_chars: list[str] = []
    for tok in tokens:
        if len(tok) == 1 and "㐀" <= tok <= "鿿":
            if ascii_buf:
                word = "".join(ascii_buf)
                if len(word) >= 2:
                    out.add(word)
                ascii_buf = []
            cjk_chars.append(tok)
        else:
            ascii_buf.append(tok)
    if ascii_buf:
        word = "".join(ascii_buf)
        if len(word) >= 2:
            out.add(word)
    if len(cjk_chars) <= 3:
        out.update(cjk_chars)
    else:
        out.update("".join(cjk_chars[i:i + 2]) for i in range(len(cjk_chars) - 1))
        out.update("".join(cjk_chars[i:i + 3]) for i in range(len(cjk_chars) - 2))
    return {t for t in out if t.strip()}


def _default_why(candidate: VisualCandidate) -> str:
    role = candidate.visual_role or "source visual"
    heading = candidate.nearby_heading or candidate.nearby_caption or "this page"
    return f"{role} on p.{candidate.page} supports {heading[:40]}"
