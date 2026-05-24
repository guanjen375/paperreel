"""LLM provider interface — outline / script / scene generation.

Local-only build: the **only** supported backend is `ollama`. There is no
mock or API-based fallback — if Ollama is not reachable, the pipeline
fails fast.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def chunk_summarize(self, chunk_text: str, *, page_range: tuple[int, int],
                        target_chars: int) -> dict:
        """Return {'summary': str, 'key_points': [str], 'headings': [str]}."""

    @abstractmethod
    def build_outline(self, chunk_summaries: list[dict], *,
                      target_minutes: float, project: str) -> dict:
        """Return a dict that validates as models.LessonOutline."""

    @abstractmethod
    def write_chapter_script(self, chapter: dict, source_pages_text: dict[int, str],
                             *, chars_per_scene: int,
                             forbid_verbatim: bool = True) -> list[dict]:
        """Return list of dicts that validate as models.ScriptScene."""


def make_llm_provider(provider_cfg: dict) -> LLMProvider:
    name = (provider_cfg or {}).get("provider", "ollama").lower()
    if name == "ollama":
        from .llm_ollama import OllamaLLM
        return OllamaLLM(provider_cfg)
    raise ValueError(
        f"unknown llm provider: {name!r} — local build only supports 'ollama'"
    )
