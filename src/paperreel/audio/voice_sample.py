"""Preprocess short local voice references for XTTS speaker_wav.

The original user-provided file is never modified. We decode it through
ffmpeg, trim only edge silence, normalize loudness, convert to mono 24 kHz
WAV, then cache the processed result under the project assets directory.
"""
from __future__ import annotations

import array
import json
import math
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..hashing import sha256_file
from ..io_utils import atomic_write_json, ensure_dir, read_json
from ..manifest import manifest_path_for


class VoiceSampleError(RuntimeError):
    """Raised when a user voice sample cannot be used safely."""


@dataclass(frozen=True)
class PreparedVoiceSample:
    original_path: str
    processed_path: str
    sha256: str
    duration_sec: float
    sample_rate_hz: int
    channels: int
    warnings: list[str]
    manifest_path: str


def _bin(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise VoiceSampleError(
            f"{name} is required to preprocess --voice-sample. "
            "Please install ffmpeg/ffprobe and try again."
        )
    return found


def _assets_voice_dir(project_paths: Any) -> Path:
    if isinstance(project_paths, dict):
        if "assets" in project_paths:
            return Path(project_paths["assets"]) / "voice"
        if "root" in project_paths:
            return Path(project_paths["root"]) / "assets" / "voice"
    return Path(project_paths) / "assets" / "voice"


def _probe_audio(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                _bin("ffprobe"),
                "-v", "error",
                "-show_streams",
                "-show_entries", "format=duration",
                "-of", "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise VoiceSampleError(
            f"Could not read voice sample audio: {path}. "
            "Use a normal WAV/MP3/M4A file with one clean speaker."
        ) from e
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise VoiceSampleError(f"ffprobe returned invalid metadata for {path}") from e

    stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"),
        None,
    )
    if not stream:
        raise VoiceSampleError(f"Voice sample has no audio stream: {path}")

    duration_raw = data.get("format", {}).get("duration") or stream.get("duration")
    try:
        duration = float(duration_raw)
    except (TypeError, ValueError) as e:
        raise VoiceSampleError(f"Could not determine voice sample duration: {path}") from e

    return {
        "duration_sec": duration,
        "sample_rate_hz": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
    }


def _duration_warnings(duration_sec: float) -> list[str]:
    warnings: list[str] = []
    if duration_sec < 4.0:
        raise VoiceSampleError(
            "--voice-sample is too short. Please provide at least 4 seconds; "
            "6-10 seconds is recommended."
        )
    if duration_sec > 20.0:
        raise VoiceSampleError(
            "--voice-sample is too long. Please keep it under 20 seconds; "
            "6-10 seconds is recommended."
        )
    if duration_sec < 6.0:
        warnings.append("voice sample is shorter than the recommended 6-10 seconds")
    elif duration_sec > 15.0:
        warnings.append("voice sample is longer than recommended; consider a 6-10 second clip")
    elif duration_sec > 10.0:
        warnings.append("voice sample is slightly longer than the recommended 6-10 seconds")
    return warnings


def _analyze_processed_wav(path: Path) -> tuple[float, int, int, list[str]]:
    try:
        with wave.open(str(path), "rb") as w:
            channels = w.getnchannels()
            sample_rate = w.getframerate()
            sample_width = w.getsampwidth()
            frames = w.readframes(w.getnframes())
            duration = w.getnframes() / max(1, sample_rate)
    except (wave.Error, OSError) as e:
        raise VoiceSampleError(f"Processed voice sample is unreadable: {path}") from e

    if sample_width != 2:
        raise VoiceSampleError("Processed voice sample must be 16-bit PCM WAV")
    if duration < 3.5:
        raise VoiceSampleError(
            "Voice sample became too short after silence trimming. "
            "Use a cleaner 6-10 second clip with continuous speech."
        )
    if duration > 20.0:
        raise VoiceSampleError("Processed voice sample is still longer than 20 seconds")

    samples = array.array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        raise VoiceSampleError("Voice sample contains no audio samples after processing")

    peak = max(abs(s) for s in samples)
    rms = math.sqrt(sum(float(s) * float(s) for s in samples) / len(samples))
    if peak < 300 or rms < 50:
        raise VoiceSampleError(
            "Voice sample is too quiet or silent. Record closer to the microphone "
            "in a quiet room and try again."
        )

    warnings: list[str] = []
    clipped = sum(1 for s in samples if abs(s) >= 32700)
    if clipped / len(samples) > 0.001:
        warnings.append("voice sample appears clipped; lower the recording gain if quality is poor")
    return duration, sample_rate, channels, warnings


def _ffmpeg_process(src: Path, dest: Path) -> None:
    ensure_dir(dest.parent)
    filters = (
        "silenceremove=start_periods=1:start_duration=0.15:start_threshold=-45dB:"
        "stop_periods=1:stop_duration=0.25:stop_threshold=-45dB,"
        "loudnorm=I=-20:TP=-2:LRA=11"
    )
    try:
        subprocess.run(
            [
                _bin("ffmpeg"),
                "-y",
                "-loglevel", "error",
                "-i", str(src),
                "-vn",
                "-af", filters,
                "-ac", "1",
                "-ar", "24000",
                "-sample_fmt", "s16",
                str(dest),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        dest.unlink(missing_ok=True)
        raise VoiceSampleError(
            f"Failed to preprocess voice sample with ffmpeg: {e.stderr.strip() or e!r}"
        ) from e


def prepare_voice_sample(
    input_path: str | Path,
    project_paths: Any,
    config: dict | None = None,
) -> PreparedVoiceSample:
    """Validate, convert, and cache a user voice sample for XTTS.

    ``config`` is accepted for future tuning; current behaviour is fixed and
    intentionally conservative.
    """
    del config
    src = Path(input_path).expanduser()
    if not src.exists():
        raise VoiceSampleError(f"Voice sample not found: {src}")
    if not src.is_file():
        raise VoiceSampleError(f"Voice sample is not a file: {src}")
    src = src.resolve()

    original_sha = sha256_file(src)
    meta = _probe_audio(src)
    warnings = _duration_warnings(float(meta["duration_sec"]))

    voice_dir = _assets_voice_dir(project_paths)
    ensure_dir(voice_dir)
    processed = voice_dir / f"reference_{original_sha[:16]}.speaker.wav"
    manifest = manifest_path_for(processed)

    if processed.exists() and manifest.exists():
        try:
            cached = read_json(manifest)
            if cached.get("original_sha256") == original_sha:
                return PreparedVoiceSample(
                    original_path=str(src),
                    processed_path=str(processed),
                    sha256=str(cached["sha256"]),
                    duration_sec=float(cached["duration_sec"]),
                    sample_rate_hz=int(cached["sample_rate_hz"]),
                    channels=int(cached["channels"]),
                    warnings=list(cached.get("warnings", [])),
                    manifest_path=str(manifest),
                )
        except Exception:
            pass

    _ffmpeg_process(src, processed)
    duration, sample_rate, channels, processed_warnings = _analyze_processed_wav(processed)
    warnings.extend(processed_warnings)
    processed_sha = sha256_file(processed)

    payload = {
        "schema": "voice_sample_v1",
        "original_path": str(src),
        "original_sha256": original_sha,
        "processed_path": str(processed),
        "sha256": processed_sha,
        "duration_sec": duration,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "warnings": warnings,
    }
    atomic_write_json(manifest, payload)
    return PreparedVoiceSample(
        original_path=str(src),
        processed_path=str(processed),
        sha256=processed_sha,
        duration_sec=duration,
        sample_rate_hz=sample_rate,
        channels=channels,
        warnings=warnings,
        manifest_path=str(manifest),
    )

