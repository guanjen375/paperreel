"""Mock TTS — emits a valid mono WAV whose length matches a CJK reading speed.

Generates a very quiet sine tone (so the resulting MP4 has real audio data
that ffmpeg can mux with AAC) rather than pure silence; this catches more
bugs than a 0-byte placeholder.
"""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from ..io_utils import ensure_dir
from ..utils.text_cleaning import cjk_char_count
from .tts_base import TTSProvider


class MockTTS(TTSProvider):
    name = "mock"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def synthesize(self, text: str, out_path: str | Path, *,
                   voice: str | None = None,
                   sample_rate_hz: int = 48000,
                   speaking_rate: float = 1.0) -> float:
        cjk = cjk_char_count(text)
        # 240 chars/min default; for very short fragments use 1.5s floor.
        seconds = max(1.5, cjk / 240.0 * 60.0 / max(0.25, speaking_rate))
        if cjk == 0:
            # ascii / mixed: count words+chars roughly
            seconds = max(1.5, len(text.split()) / 2.8 + len(text) / 80.0)
        out = Path(out_path)
        ensure_dir(out.parent)
        n_samples = int(seconds * sample_rate_hz)
        amp = 1000        # very quiet, well under int16 max (32767)
        freq = 220.0      # A3 hum
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate_hz)
            # Write in chunks to avoid building a huge list.
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
