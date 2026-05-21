"""Anthropic provider stub. Activated when llm.provider=='anthropic' AND the
`anthropic` extra is installed AND ANTHROPIC_API_KEY is set.

This is intentionally minimal in the MVP: it sends the same prompt shape that
the MockLLM understands and falls back to MockLLM if any precondition fails.
The structured prompts (繁體中文, 不逐字照抄, 保留頁碼) are enforced here.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .llm_base import LLMProvider
from .llm_mock import MockLLM


SYSTEM_PROMPT = (
    "你是一位繁體中文教學影片腳本作者。"
    "嚴格遵守以下規則："
    "1. 只用繁體中文輸出，不要使用簡體字。"
    "2. 不可逐字照抄輸入內容，請以摘要、轉述、教學化的方式重寫。"
    "3. 每段必須保留來源頁碼。"
    "4. 回傳結構化 JSON，欄位與使用者要求一致；不要多餘文字。"
)


class AnthropicLLM(LLMProvider):
    name = "anthropic"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self._client: Any | None = None
        self._fallback = MockLLM(cfg)
        self.model = self.cfg.get("model", "claude-opus-4-7")
        self.temperature = float(self.cfg.get("temperature", 0.4))

    def _ensure_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        if "ANTHROPIC_API_KEY" not in os.environ:
            return None
        try:
            import anthropic  # type: ignore
        except ImportError:
            return None
        self._client = anthropic.Anthropic()
        return self._client

    # ---------- private helpers ----------

    def _ask_json(self, user_prompt: str, *, max_tokens: int = 4000) -> dict | list | None:
        client = self._ensure_client()
        if client is None:
            return None
        try:
            msg = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = "".join(
                getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text"
            )
            text = text.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lstrip().lower().startswith("json"):
                    text = text.split("\n", 1)[1] if "\n" in text else text
            return json.loads(text)
        except Exception:
            return None

    # ---------- interface ----------
    def chunk_summarize(self, chunk_text: str, *, page_range: tuple[int, int],
                        target_chars: int) -> dict:
        prompt = (
            f"以下是 PDF 第 {page_range[0]}–{page_range[1]} 頁原文（可能含 OCR 雜訊）。\n"
            "請輸出 JSON：{\"summary\": <2~3 句摘要>, "
            "\"key_points\": [<最多 5 個學習重點>], "
            "\"headings\": [<觀察到的小節標題>]}。\n"
            f"摘要長度上限 {target_chars} 字。\n"
            "---\n"
            f"{chunk_text[:12000]}"
        )
        out = self._ask_json(prompt, max_tokens=1200)
        if isinstance(out, dict):
            out.setdefault("page_range", list(page_range))
            return out
        return self._fallback.chunk_summarize(
            chunk_text, page_range=page_range, target_chars=target_chars,
        )

    def build_outline(self, chunk_summaries: list[dict], *,
                      target_minutes: float, project: str) -> dict:
        prompt = (
            f"專案名: {project}\n"
            f"目標影片長度: {target_minutes:.1f} 分鐘\n"
            "以下是各段摘要 (JSON list)。請輸出 lesson outline JSON，"
            "結構如下：\n"
            "{\"project\":..., \"language\":\"zh-TW\", \"target_minutes\":..., "
            "\"rationale\":..., \"chapters\":[{\"chapter_id\":..., \"title\":..., "
            "\"source_pages\":[..], \"target_minutes\":..., "
            "\"key_points\":[..], \"recap\":bool}]}\n"
            "---\n"
            + json.dumps(chunk_summaries, ensure_ascii=False)
        )
        out = self._ask_json(prompt, max_tokens=3000)
        if isinstance(out, dict) and "chapters" in out:
            return out
        return self._fallback.build_outline(
            chunk_summaries, target_minutes=target_minutes, project=project,
        )

    def write_chapter_script(self, chapter: dict, source_pages_text: dict[int, str],
                             *, chars_per_scene: int,
                             forbid_verbatim: bool = True) -> list[dict]:
        source_blob = "\n\n".join(
            f"[p.{p}]\n{source_pages_text.get(p, '')[:1800]}"
            for p in chapter.get("source_pages", [])
        )
        prompt = (
            f"章節資料: {json.dumps(chapter, ensure_ascii=False)}\n"
            f"每個 scene 旁白上限 {chars_per_scene} 個字。\n"
            "請輸出 JSON list，每筆為 ScriptScene："
            "{\"scene_id\":..., \"chapter_id\":..., \"title\":..., "
            "\"source_pages\":[..], \"source_refs\":[..], "
            "\"narration_text_zh_tw\":..., \"on_screen_text\":..., "
            "\"visual_hint\":..., \"visual_type\":\"bullet_card|title_card|diagram|recap|quiz\", "
            "\"estimated_duration_sec\":..}\n"
            "嚴禁逐字照抄；每 scene 至少標註 1 個 source_page。\n"
            "--- 原始素材 ---\n"
            + source_blob
        )
        out = self._ask_json(prompt, max_tokens=4000)
        if isinstance(out, list) and out:
            return out
        return self._fallback.write_chapter_script(
            chapter, source_pages_text,
            chars_per_scene=chars_per_scene, forbid_verbatim=forbid_verbatim,
        )
