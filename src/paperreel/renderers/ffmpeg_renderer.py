"""ffmpeg subprocess helpers.

The renderer never inlines a Python ffmpeg lib — it just shells out so the
user controls the binary version (e.g. the one matching their CUDA / VAAPI).
All calls are wrapped so that a missing binary becomes a clear `FfmpegMissing`
error rather than a noisy traceback.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Iterable

from ..io_utils import ensure_dir


class FfmpegMissing(RuntimeError):
    pass


class FfmpegError(RuntimeError):
    def __init__(self, returncode: int, cmd: list[str], stderr: str):
        super().__init__(f"ffmpeg exit {returncode}: {stderr.strip()[:600]}")
        self.returncode = returncode
        self.cmd = cmd
        self.stderr = stderr


def have_ffmpeg(binary: str = "ffmpeg") -> bool:
    return shutil.which(binary) is not None


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    if shutil.which(cmd[0]) is None:
        raise FfmpegMissing(f"binary not found on PATH: {cmd[0]}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FfmpegError(proc.returncode, cmd, proc.stderr)
    return proc


def probe_duration_seconds(media: str | Path, *, binary: str = "ffprobe") -> float:
    p = Path(media)
    if p.suffix.lower() == ".wav":
        try:
            with wave.open(str(p), "rb") as w:
                return w.getnframes() / max(1, w.getframerate())
        except Exception:
            pass
    cmd = [
        binary, "-v", "error", "-of", "json",
        "-show_entries", "format=duration", str(p),
    ]
    proc = _run(cmd)
    data = json.loads(proc.stdout or "{}")
    return float(data.get("format", {}).get("duration", 0.0))


def render_still_with_audio(
    image_path: str | Path,
    audio_path: str | Path,
    out_path: str | Path,
    *,
    duration_sec: float,
    fps: int = 30,
    resolution: tuple[int, int] = (1920, 1080),
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    pixel_format: str = "yuv420p",
    ken_burns: bool = False,
    ffmpeg_binary: str = "ffmpeg",
) -> str:
    """Mux a single still image with an audio track into an MP4 segment.

    If `ken_burns` is True the image is gently zoomed in over the full
    duration via the `zoompan` filter, which gives the visual some motion
    even when the underlying asset is a static card.
    """
    out = Path(out_path)
    ensure_dir(out.parent)
    w, h = resolution
    if ken_burns:
        frames = max(1, int(round(duration_sec * fps)))
        vf = (
            f"zoompan=z='min(zoom+0.0008,1.12)':d={frames}"
            f":s={w}x{h}:fps={fps},format={pixel_format}"
        )
    else:
        vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,"\
             f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,format={pixel_format}"
    cmd = [
        ffmpeg_binary, "-y", "-loglevel", "error",
        "-loop", "1", "-i", str(image_path),
        "-i", str(audio_path),
        "-t", f"{duration_sec:.3f}",
        "-vf", vf,
        "-r", str(fps),
        "-c:v", video_codec,
        "-c:a", audio_codec, "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(out),
    ]
    _run(cmd)
    return str(out)


def concat_segments(
    segment_paths: Iterable[str | Path],
    out_path: str | Path,
    *,
    ffmpeg_binary: str = "ffmpeg",
) -> str:
    """Concatenate MP4 segments using the concat demuxer (no re-encode)."""
    out = Path(out_path)
    ensure_dir(out.parent)
    list_file = out.with_suffix(".concat.txt")
    lines: list[str] = []
    seg_count = 0
    for s in segment_paths:
        sp = Path(s)
        if not sp.exists():
            raise FileNotFoundError(f"segment missing: {sp}")
        # ffmpeg concat list expects forward slashes + single-quoted paths.
        lines.append(f"file '{sp.resolve().as_posix()}'\n")
        seg_count += 1
    if seg_count == 0:
        raise ValueError("no segments to concat")
    list_file.write_text("".join(lines), encoding="utf-8")
    cmd = [
        ffmpeg_binary, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        "-movflags", "+faststart",
        str(out),
    ]
    _run(cmd)
    return str(out)
