"""Edge TTS provider — optional, falls back to MockTTS when unavailable.

Requires the `edge-tts` extra: pip install pdf2lesson-video-tw[edge]
Outputs WAV via ffmpeg post-processing (edge-tts produces mp3 by default).
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

from ..io_utils import ensure_dir
from .tts_base import TTSProvider
from .tts_mock import MockTTS


class EdgeTTS(TTSProvider):
    name = "edge"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self._fallback = MockTTS(cfg)

    def synthesize(self, text: str, out_path: str | Path, *,
                   voice: str | None = None,
                   sample_rate_hz: int = 48000,
                   speaking_rate: float = 1.0) -> float:
        try:
            import edge_tts  # type: ignore
        except ImportError:
            return self._fallback.synthesize(
                text, out_path,
                voice=voice, sample_rate_hz=sample_rate_hz,
                speaking_rate=speaking_rate,
            )
        if shutil.which("ffmpeg") is None:
            return self._fallback.synthesize(
                text, out_path,
                voice=voice, sample_rate_hz=sample_rate_hz,
                speaking_rate=speaking_rate,
            )
        v = voice or self.cfg.get("voice", "zh-TW-HsiaoChenNeural")
        rate_pct = int((speaking_rate - 1.0) * 100)
        rate_str = f"{rate_pct:+d}%"
        out = Path(out_path)
        ensure_dir(out.parent)
        mp3_tmp = out.with_suffix(".edge.mp3")

        async def _run() -> None:
            communicator = edge_tts.Communicate(text=text, voice=v, rate=rate_str)
            await communicator.save(str(mp3_tmp))

        try:
            asyncio.run(_run())
            # transcode to wav with target sample rate
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", str(mp3_tmp),
                 "-ar", str(sample_rate_hz),
                 "-ac", "1",
                 str(out)],
                check=True,
            )
            mp3_tmp.unlink(missing_ok=True)
            return _wav_duration_seconds(out)
        except Exception:
            mp3_tmp.unlink(missing_ok=True)
            return self._fallback.synthesize(
                text, out_path,
                voice=voice, sample_rate_hz=sample_rate_hz,
                speaking_rate=speaking_rate,
            )


def _wav_duration_seconds(path: Path) -> float:
    import wave
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
    return frames / max(1, rate)
