"""TTS provider interface — all providers output mono WAV at the
project's target sample rate.

Local-only build: the **only** supported backend is `xtts` (Coqui XTTS v2).
"""
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
    name = (cfg or {}).get("provider", "xtts").lower()
    if name == "xtts":
        from .tts_xtts import XttsTTS
        return XttsTTS(cfg)
    raise ValueError(
        f"unknown tts provider: {name!r} — local build only supports 'xtts'"
    )
