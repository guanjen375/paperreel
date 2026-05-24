"""Ollama LLM provider — pure local inference via Ollama's HTTP API.

This is the **only** LLM backend in the local build. There is no mock
fallback: if Ollama can't be reached or the model isn't pulled, the
pipeline fails fast so the user notices and fixes the setup, instead of
silently producing placeholder output.

Defaults talk to ``http://localhost:11434`` and use ``qwen2.5:14b-instruct``
(decent繁中 quality, fits on a single 16 GB GPU after Q4 quantisation).
Override via ``llm.base_url`` / ``llm.model`` in the config.

Requires the ``ollama`` extra: ``pip install paperreel[ollama]``.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .llm_base import LLMProvider


SYSTEM_PROMPT = (
    "你是一位繁體中文教學影片腳本作者。"
    "嚴格遵守以下規則："
    "1. 只用繁體中文輸出，不要使用簡體字。"
    "2. 不可逐字照抄輸入內容，請以摘要、轉述、教學化的方式重寫。"
    "3. 每段必須保留來源頁碼。"
    "4. 回傳結構化 JSON，欄位與使用者要求一致；不要多餘文字、不要 markdown 程式區塊。"
)

# Match the first {...} or [...] block in case the model wraps JSON in prose.
_JSON_BLOCK = re.compile(r"(\{[\s\S]*\}|\[[\s\S]*\])")


class OllamaUnavailable(RuntimeError):
    """Raised when the Ollama daemon / model is not usable."""


class OllamaLLM(LLMProvider):
    name = "ollama"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.base_url = str(self.cfg.get("base_url", "http://localhost:11434")).rstrip("/")
        self.model = str(self.cfg.get("model", "qwen2.5:14b-instruct"))
        self.temperature = float(self.cfg.get("temperature", 0.4))
        self.num_ctx = int(self.cfg.get("num_ctx", 8192))
        self.request_timeout_sec = float(self.cfg.get("request_timeout_sec", 600))
        self._client: Any | None = None

    # ---------- transport ----------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import ollama  # type: ignore
        except ImportError as e:
            raise OllamaUnavailable(
                "ollama package not installed — run: pip install -e \".[ollama]\""
            ) from e
        self._client = ollama.Client(host=self.base_url, timeout=self.request_timeout_sec)
        # Touch the server so we fail fast (and with a useful message) instead
        # of later, mid-pipeline.
        try:
            self._client.list()
        except Exception as e:  # network down, daemon not running, …
            raise OllamaUnavailable(
                f"cannot reach Ollama at {self.base_url}: {e!r}\n"
                "  start it with `ollama serve` then `ollama pull "
                f"{self.model}`"
            ) from e
        return self._client

    def _ask_json(self, user_prompt: str, *, max_tokens: int = 4000) -> dict | list:
        client = self._ensure_client()
        try:
            resp = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                    "num_predict": max_tokens,
                },
            )
        except Exception as e:
            raise OllamaUnavailable(
                f"Ollama chat call failed ({self.model}): {e!r}"
            ) from e
        text = (resp.get("message") or {}).get("content", "").strip()
        if not text:
            raise OllamaUnavailable(f"Ollama returned empty response for {self.model}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = _JSON_BLOCK.search(text)
            if not m:
                raise OllamaUnavailable(
                    f"Ollama did not return valid JSON: {text[:200]}…"
                )
            return json.loads(m.group(1))

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
        if not isinstance(out, dict):
            raise OllamaUnavailable(
                f"chunk_summarize: expected dict, got {type(out).__name__}"
            )
        out.setdefault("page_range", list(page_range))
        out.setdefault("summary", "")
        out.setdefault("key_points", [])
        out.setdefault("headings", [])
        return out

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
            "chapter_id 必須是 ch_001 / ch_002 這種格式，target_minutes 加總接近目標長度。\n"
            "---\n"
            + json.dumps(chunk_summaries, ensure_ascii=False)
        )
        out = self._ask_json(prompt, max_tokens=3000)
        if not isinstance(out, dict) or "chapters" not in out:
            raise OllamaUnavailable(
                "build_outline: response missing 'chapters' field"
            )
        out.setdefault("project", project)
        out.setdefault("language", "zh-TW")
        out.setdefault("target_minutes", float(target_minutes))
        out.setdefault("rationale", f"ollama:{self.model}")
        return out

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
            "請輸出 JSON 物件，格式為 {\"scenes\": [...]}，scenes 每筆為 ScriptScene："
            "{\"scene_id\":..., \"chapter_id\":..., \"title\":..., "
            "\"source_pages\":[..], \"source_refs\":[..], "
            "\"narration_text_zh_tw\":..., \"on_screen_text\":..., "
            "\"visual_hint\":..., "
            "\"visual_type\":\"bullet_card|title_card|diagram|recap|quiz|generated_image\", "
            "\"estimated_duration_sec\":..}\n"
            "規則：\n"
            "- scene_id 用 <chapter_id>_sc_001 / _sc_002 順序編號。\n"
            "- 嚴禁逐字照抄原文；每 scene 至少標註 1 個 source_page。\n"
            "- visual_hint 應該是給圖片產生器的英文描述 (例如 'minimalist diagram of...');"
            "  visual_type=generated_image 才會實際呼叫 SDXL,其他類型用簡報卡片。\n"
            "- on_screen_text 是投影片上要顯示的中文短句（最多 30 字），可換行。\n"
            "--- 原始素材 ---\n"
            + source_blob
        )
        out = self._ask_json(prompt, max_tokens=4000)
        if isinstance(out, list):
            scenes = out
        elif isinstance(out, dict):
            scenes = out.get("scenes") or out.get("script") or []
        else:
            raise OllamaUnavailable(
                f"write_chapter_script: unexpected response type {type(out).__name__}"
            )
        if not isinstance(scenes, list) or not scenes:
            raise OllamaUnavailable(
                "write_chapter_script: model returned no scenes"
            )
        return scenes
