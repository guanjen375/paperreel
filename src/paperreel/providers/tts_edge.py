"""Edge TTS provider.

Requires the `edge-tts` extra: `pip install -e ".[edge]"` and ffmpeg on PATH.
Outputs WAV via ffmpeg post-processing (edge-tts produces mp3 by default).

Missing package, missing ffmpeg, or synthesis failure all raise loudly — the
pipeline no longer silently degrades to a placeholder.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

from ..io_utils import ensure_dir
from .tts_base import TTSProvider


class EdgeTTSError(RuntimeError):
    """Raised when the Edge TTS provider cannot satisfy a request."""


class EdgeTTS(TTSProvider):
    name = "edge"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def synthesize(self, text: str, out_path: str | Path, *,
                   voice: str | None = None,
                   sample_rate_hz: int = 48000,
                   speaking_rate: float = 1.0) -> float:
        try:
            import edge_tts  # type: ignore
        except ImportError as e:
            raise EdgeTTSError(
                "edge-tts package not installed — run "
                "`pip install -e \".[edge]\"`"
            ) from e
        if shutil.which("ffmpeg") is None:
            raise EdgeTTSError("ffmpeg not on PATH — required to transcode edge-tts mp3 → wav")
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
        except Exception as e:
            mp3_tmp.unlink(missing_ok=True)
            raise EdgeTTSError(f"edge-tts synthesis failed: {e!r}") from e


def _wav_duration_seconds(path: Path) -> float:
    import wave
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
    return frames / max(1, rate)
