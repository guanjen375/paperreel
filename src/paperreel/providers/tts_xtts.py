"""Coqui XTTS v2 TTS provider — local, GPU-accelerated, zero network.

Model: `tts_models/multilingual/multi-dataset/xtts_v2` (Coqui).

Default behaviour:
- Language is forced to `zh-cn` (XTTS uses the zh-cn tag; zh-tw audio is
  produced by feeding it 繁中 text and the same code).
- Voice is selected by either:
  1. `tts.speaker_wav`  → absolute path to a 6–10s reference WAV (best 質感)
  2. `tts.speaker`      → one of XTTS's built-in speaker names
  3. neither             → falls back to a sensible default built-in
- Output is written at the model's native sample rate, then transcoded to
  the project's target sample rate with ffmpeg.

There is no mock fallback: if the TTS package or model is unavailable,
this raises ``XttsUnavailable`` and the pipeline fails loudly.

Install: ``pip install paperreel[xtts]``  (which pulls torch + TTS).
The first call downloads ~1.8 GB of weights to ``~/.local/share/tts``.
"""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

from ..io_utils import ensure_dir
from .tts_base import TTSProvider


XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_SPEAKER = "Ana Florence"   # 中性女聲, XTTS 內建


class XttsUnavailable(RuntimeError):
    """Raised when Coqui XTTS cannot be loaded or used."""


def _ffmpeg_bin() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / max(1, w.getframerate())


class XttsTTS(TTSProvider):
    name = "xtts"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.speaker = self.cfg.get("speaker") or DEFAULT_SPEAKER
        self.speaker_wav = self.cfg.get("speaker_wav")
        self.language = str(self.cfg.get("language", "zh-cn"))
        self.device = str(self.cfg.get("device", "auto"))  # auto|cuda|cpu
        self._tts: Any | None = None
        self._loaded_device: str | None = None

    # ---------- model loading (lazy, expensive) ----------

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # type: ignore
        except ImportError as e:
            raise XttsUnavailable(
                "torch not installed — run: pip install -e \".[xtts]\""
            ) from e
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_model(self) -> Any:
        if self._tts is not None:
            return self._tts
        try:
            from TTS.api import TTS  # type: ignore
        except ImportError as e:
            raise XttsUnavailable(
                "Coqui TTS not installed — run: pip install -e \".[xtts]\""
            ) from e
        device = self._resolve_device()
        try:
            tts = TTS(XTTS_MODEL)
            tts = tts.to(device)
        except Exception as e:
            raise XttsUnavailable(
                f"failed to load {XTTS_MODEL} on {device}: {e!r}\n"
                "  first run will download ~1.8 GB to ~/.local/share/tts"
            ) from e
        self._tts = tts
        self._loaded_device = device
        return tts

    # ---------- TTSProvider interface ----------

    def synthesize(self, text: str, out_path: str | Path, *,
                   voice: str | None = None,
                   sample_rate_hz: int = 48000,
                   speaking_rate: float = 1.0) -> float:
        if not text or not text.strip():
            raise XttsUnavailable("xtts: empty text — refusing to synthesize silence")

        out = Path(out_path)
        ensure_dir(out.parent)
        tts = self._ensure_model()

        # XTTS writes 24 kHz wav natively; transcode to target SR below.
        raw_wav = out.with_suffix(".xtts.raw.wav")

        speaker_kwargs: dict[str, Any] = {}
        if self.speaker_wav:
            speaker_kwargs["speaker_wav"] = str(self.speaker_wav)
        else:
            # voice param (per-call) wins over config-level speaker.
            speaker_kwargs["speaker"] = voice or self.speaker

        try:
            tts.tts_to_file(
                text=text,
                language=self.language,
                file_path=str(raw_wav),
                split_sentences=True,
                **speaker_kwargs,
            )
        except Exception as e:
            raw_wav.unlink(missing_ok=True)
            raise XttsUnavailable(f"xtts synth failed: {e!r}") from e

        # Apply speaking_rate via ffmpeg atempo and resample.
        atempo = max(0.5, min(2.0, float(speaking_rate)))  # ffmpeg atempo range
        try:
            subprocess.run(
                [_ffmpeg_bin(), "-y", "-loglevel", "error",
                 "-i", str(raw_wav),
                 "-filter:a", f"atempo={atempo}",
                 "-ar", str(int(sample_rate_hz)),
                 "-ac", "1",
                 str(out)],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raw_wav.unlink(missing_ok=True)
            raise XttsUnavailable(
                f"ffmpeg post-processing failed (ffmpeg on PATH?): {e!r}"
            ) from e
        raw_wav.unlink(missing_ok=True)
        return _wav_duration_seconds(out)
