"""Fake LLM used only inside the test suite.

Deterministic and offline — picks short paraphrased fragments from the
source so the verbatim-detection / source-provenance paths still get
exercised, but labels itself in `rationale` so a real run could never be
confused with a fake one.
"""
from __future__ import annotations

import re

from paperreel.providers.llm_base import LLMProvider
from paperreel.utils.text_cleaning import (cjk_char_count, extract_headings,
                                            normalise_text)

_SENT_SPLIT = re.compile(r"(?<=[。．.!?！？])\s*")


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    # ---------- chunk summary ----------
    def chunk_summarize(self, chunk_text: str, *, page_range: tuple[int, int],
                        target_chars: int) -> dict:
        text = normalise_text(chunk_text)
        headings = extract_headings(text, max_lines=80)
        sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
        picks: list[str] = []
        if sents:
            picks.append(sents[0])
        if len(sents) > 4:
            picks.append(sents[len(sents) // 2])
        if len(sents) > 1:
            picks.append(sents[-1])
        summary = " ".join(picks)[: max(120, target_chars)]
        key_points = [
            (h if len(h) <= 40 else h[:40] + "…") for h in headings[:5]
        ] or [f"重點摘要 (頁 {page_range[0]}–{page_range[1]})"]
        return {
            "summary": summary or f"(第 {page_range[0]}–{page_range[1]} 頁摘要)",
            "key_points": key_points,
            "headings": headings,
            "page_range": list(page_range),
        }

    # ---------- outline ----------
    def build_outline(self, chunk_summaries: list[dict], *,
                      target_minutes: float, project: str) -> dict:
        chapters: list[dict] = []
        if not chunk_summaries:
            chapters.append({
                "chapter_id": "ch_001",
                "title": f"{project} 概覽",
                "source_pages": [1],
                "target_minutes": float(target_minutes),
                "key_points": [],
                "recap": False,
            })
        else:
            n = max(1, min(len(chunk_summaries), max(1, round(target_minutes / 8.0))))
            per = max(1, len(chunk_summaries) // n)
            for i in range(n):
                bucket = chunk_summaries[i * per: (i + 1) * per if i < n - 1 else len(chunk_summaries)]
                if not bucket:
                    continue
                pages = sorted({p for cs in bucket for p in (cs.get("page_range") or [])})
                headings = [h for cs in bucket for h in cs.get("headings", [])]
                title = headings[0] if headings else f"第 {i + 1} 章 — 重點"
                chapters.append({
                    "chapter_id": f"ch_{i + 1:03d}",
                    "title": title[:40],
                    "source_pages": pages or [1],
                    "target_minutes": round(float(target_minutes) / max(1, n), 2),
                    "key_points": [b["summary"][:60] for b in bucket[:3]],
                    "recap": i > 0 and (i % 2 == 0),
                })
        return {
            "project": project,
            "language": "zh-TW",
            "target_minutes": float(target_minutes),
            "rationale": "fake outline: even-bin by heading count",
            "chapters": chapters,
        }

    # ---------- script ----------
    def write_chapter_script(self, chapter: dict, source_pages_text: dict[int, str],
                             *, chars_per_scene: int,
                             forbid_verbatim: bool = True) -> list[dict]:
        pages = list(chapter.get("source_pages") or [1])
        scenes: list[dict] = []
        per_scene_pages = max(1, len(pages) // max(1, round(chapter.get("target_minutes", 5) / 1.0)))
        idx = 0
        scene_no = 1
        while idx < len(pages):
            sp = pages[idx: idx + per_scene_pages]
            idx += per_scene_pages
            joined = " ".join(
                normalise_text(source_pages_text.get(p, ""))[: max(60, chars_per_scene // 2)]
                for p in sp
            ).strip()
            if not joined:
                joined = f"本段聚焦於第 {sp[0]} 頁的重點。"
            base = joined.split("。")[0][:max(40, chars_per_scene - 60)]
            narration = (
                f"接下來這段課程，我們聚焦在第 {sp[0]} 頁附近的核心觀念。"
                f"你可以這樣理解：{base}。"
                f"請記住這個重點，我們稍後會再回顧。"
            )[:chars_per_scene]
            on_screen = (chapter.get("title") or "重點")[:30]
            scenes.append({
                "scene_id": f"{chapter['chapter_id']}_sc_{scene_no:03d}",
                "chapter_id": chapter["chapter_id"],
                "title": f"{chapter.get('title', '重點')}：第 {scene_no} 段",
                "source_pages": sp,
                "source_refs": [f"p.{p}" for p in sp],
                "narration_text_zh_tw": narration,
                "on_screen_text": on_screen,
                "visual_hint": "簡報式重點卡",
                "visual_type": "bullet_card",
                "estimated_duration_sec": max(20.0,
                                              cjk_char_count(narration) / 240.0 * 60.0),
            })
            scene_no += 1
        if not scenes:
            scenes.append({
                "scene_id": f"{chapter['chapter_id']}_sc_001",
                "chapter_id": chapter["chapter_id"],
                "title": chapter.get("title", "重點"),
                "source_pages": pages or [1],
                "source_refs": [f"p.{p}" for p in (pages or [1])],
                "narration_text_zh_tw": "（測試佔位旁白）這一段是測試用 fake 內容。",
                "on_screen_text": "重點",
                "visual_hint": "簡報式重點卡",
                "visual_type": "bullet_card",
                "estimated_duration_sec": 25.0,
            })
        return scenes
