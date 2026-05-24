"""Test-only fake TTS — emits a quiet sine wav whose length tracks CJK count."""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from paperreel.io_utils import ensure_dir
from paperreel.providers.tts_base import TTSProvider
from paperreel.utils.text_cleaning import cjk_char_count


class FakeTTS(TTSProvider):
    name = "fake"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def synthesize(self, text: str, out_path: str | Path, *,
                   voice: str | None = None,
                   sample_rate_hz: int = 48000,
                   speaking_rate: float = 1.0) -> float:
        cjk = cjk_char_count(text)
        seconds = max(1.5, cjk / 240.0 * 60.0 / max(0.25, speaking_rate))
        if cjk == 0:
            seconds = max(1.5, len(text.split()) / 2.8 + len(text) / 80.0)
        out = Path(out_path)
        ensure_dir(out.parent)
        n_samples = int(seconds * sample_rate_hz)
        amp = 1000
        freq = 220.0
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate_hz)
            chunk = 4096
            two_pi_f = 2 * math.pi * freq / sample_rate_hz
            for start in range(0, n_samples, chunk):
                end = min(n_samples, start + chunk)
                frames = b"".join(
                    struct.pack("<h", int(amp * math.sin(two_pi_f * i)))
                    for i in range(start, end)
                )
                w.writeframes(frames)
        return seconds
