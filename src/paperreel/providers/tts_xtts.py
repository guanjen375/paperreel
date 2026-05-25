"""Coqui XTTS v2 TTS provider — local, GPU-accelerated, zero network.

Model: `tts_models/multilingual/multi-dataset/xtts_v2` (Coqui).

Default behaviour:
- Language is forced to `zh-cn` (XTTS uses the zh-cn tag; zh-tw audio is
  produced by feeding it 繁中 text and the same code).
- Voice is selected by either:
  1. `--voice-sample` / `tts.voice_sample` → validated local user clip,
     preprocessed into `tts.speaker_wav` by the CLI
  2. `tts.speaker_wav`  → advanced: absolute path to a processed reference WAV
  3. `tts.speaker`      → one of XTTS's built-in speaker names
  4. neither            → falls back to a sensible default built-in
- Output is written at the model's native sample rate, then transcoded to
  the project's target sample rate with ffmpeg.

There is no mock fallback: if the TTS package or model is unavailable,
this raises ``XttsUnavailable`` and the pipeline fails loudly.

Install: ``pip install -e ".[xtts]"`` (pulls torch + the maintained
``coqui-tts`` fork; the original ``TTS`` package is Python <3.12 only).
The first call downloads ~1.8 GB of weights to ``~/.local/share/tts``.
Set ``COQUI_TOS_AGREED=1`` so the first download doesn't block on the
interactive CPML license prompt.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

from ..io_utils import ensure_dir
from .tts_base import TTSProvider


XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_SPEAKER = "Ana Florence"   # 中性女聲, XTTS 內建
_XTTS_ZH_SAFE_CHARS = 80
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])\s*")
_CLAUSE_BOUNDARY = re.compile(r"(?<=[，、,：:])\s*")


class XttsUnavailable(RuntimeError):
    """Raised when Coqui XTTS cannot be loaded or used."""


def _ffmpeg_bin() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / max(1, w.getframerate())


def _split_xtts_text(text: str, *, max_chars: int = _XTTS_ZH_SAFE_CHARS
                     ) -> list[str]:
    """Split text before Coqui XTTS sees it.

    XTTS warns and may truncate Chinese text above roughly 82 characters
    even with ``split_sentences=True``. We split deterministically at
    sentence/clause boundaries, then hard-wrap any remaining long clause.
    """
    clean = re.sub(r"\s+", " ", text.strip())
    if not clean:
        return []

    pieces: list[str] = []
    for sentence in _SENTENCE_BOUNDARY.split(clean):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        for clause in _CLAUSE_BOUNDARY.split(sentence):
            clause = clause.strip()
            if not clause:
                continue
            while len(clause) > max_chars:
                pieces.append(clause[:max_chars])
                clause = clause[max_chars:].strip()
            if clause:
                pieces.append(clause)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + len(piece) <= max_chars:
            current += piece
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _concat_file_line(path: Path) -> str:
    escaped = str(path).replace("'", "'\\''")
    return f"file '{escaped}'"


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

        speaker_kwargs: dict[str, Any] = {}
        if self.speaker_wav:
            speaker_kwargs["speaker_wav"] = str(self.speaker_wav)
        else:
            # voice param (per-call) wins over config-level speaker.
            speaker_kwargs["speaker"] = voice or self.speaker

        chunks = _split_xtts_text(text)
        if not chunks:
            raise XttsUnavailable("xtts: empty text after normalisation")

        raw_parts: list[Path] = []
        concat_list = out.with_suffix(".xtts.concat.txt")
        concat_wav = out.with_suffix(".xtts.concat.raw.wav")
        source_wav: Path | None = None
        try:
            for idx, chunk in enumerate(chunks):
                raw_part = out.with_suffix(f".xtts.{idx:03d}.raw.wav")
                tts.tts_to_file(
                    text=chunk,
                    language=self.language,
                    file_path=str(raw_part),
                    split_sentences=False,
                    **speaker_kwargs,
                )
                raw_parts.append(raw_part)

            if len(raw_parts) == 1:
                source_wav = raw_parts[0]
            else:
                concat_list.write_text(
                    "\n".join(_concat_file_line(p) for p in raw_parts) + "\n"
                )
                subprocess.run(
                    [_ffmpeg_bin(), "-y", "-loglevel", "error",
                     "-f", "concat", "-safe", "0",
                     "-i", str(concat_list),
                     str(concat_wav)],
                    check=True,
                )
                source_wav = concat_wav

            # Apply speaking_rate via ffmpeg atempo and resample.
            atempo = max(0.5, min(2.0, float(speaking_rate)))
            subprocess.run(
                [_ffmpeg_bin(), "-y", "-loglevel", "error",
                 "-i", str(source_wav),
                 "-filter:a", f"atempo={atempo}",
                 "-ar", str(int(sample_rate_hz)),
                 "-ac", "1",
                 str(out)],
                check=True,
            )
        except Exception as e:
            out.unlink(missing_ok=True)
            raise XttsUnavailable(f"xtts synth failed: {e!r}") from e
        finally:
            concat_list.unlink(missing_ok=True)
            concat_wav.unlink(missing_ok=True)
            for raw_part in raw_parts:
                raw_part.unlink(missing_ok=True)

        return _wav_duration_seconds(out)
