"""TTS provider interface — all providers output 48kHz mono wav by default."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TTSProvider(ABC):
    name: str = "base"

    @abstractmethod
    def synthesize(self, text: str, out_path: str | Path, *,
                   voice: str | None = None,
                   sample_rate_hz: int = 48000,
                   speaking_rate: float = 1.0) -> float:
        """Render `text` to `out_path` (wav).

        Returns: actual duration in seconds.
        """


def make_tts_provider(cfg: dict) -> TTSProvider:
    name = (cfg or {}).get("provider", "edge").lower()
    if name == "edge":
        from .tts_edge import EdgeTTS
        return EdgeTTS(cfg)
    raise ValueError(f"unknown tts provider: {name}")
