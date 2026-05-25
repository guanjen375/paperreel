from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from paperreel.audio.voice_sample import VoiceSampleError, prepare_voice_sample
from paperreel.manifest import read_manifest
from paperreel.models import Scene, SceneStatus, VisualType
from paperreel.stages.synthesize_audio import _audio_input_hash


HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _write_wav(path: Path, *, seconds: float, rate: int = 16000,
               channels: int = 1, freq: float = 220.0, amp: int = 6000) -> None:
    n_samples = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        for i in range(n_samples):
            sample = int(amp * math.sin(2 * math.pi * freq * i / rate))
            frame = struct.pack("<h", sample) * channels
            w.writeframes(frame)


def test_missing_voice_sample_path_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(VoiceSampleError, match="not found"):
        prepare_voice_sample(tmp_path / "missing.wav", {"assets": tmp_path / "assets"})


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_too_short_voice_sample_fails(tmp_path: Path) -> None:
    sample = tmp_path / "short.wav"
    _write_wav(sample, seconds=2.0)
    with pytest.raises(VoiceSampleError, match="too short"):
        prepare_voice_sample(sample, {"assets": tmp_path / "assets"})


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_too_long_voice_sample_fails(tmp_path: Path) -> None:
    sample = tmp_path / "long.wav"
    _write_wav(sample, seconds=21.0)
    with pytest.raises(VoiceSampleError, match="too long"):
        prepare_voice_sample(sample, {"assets": tmp_path / "assets"})


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_valid_voice_sample_converts_to_mono_manifested_wav(tmp_path: Path) -> None:
    sample = tmp_path / "voice.wav"
    _write_wav(sample, seconds=6.2, rate=16000, channels=2)

    prepared = prepare_voice_sample(sample, {"assets": tmp_path / "assets"})

    assert Path(prepared.processed_path).exists()
    assert prepared.processed_path.endswith(".speaker.wav")
    assert prepared.channels == 1
    assert prepared.sample_rate_hz == 24000
    assert 4.0 <= prepared.duration_sec <= 7.0
    manifest = read_manifest(prepared.processed_path)
    assert manifest is not None
    assert manifest["schema"] == "voice_sample_v1"
    assert manifest["sha256"] == prepared.sha256
    assert manifest["duration_sec"] == prepared.duration_sec
    assert manifest["processed_path"] == prepared.processed_path


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_audio_cache_hash_changes_when_processed_voice_sample_changes(tmp_path: Path) -> None:
    sample_a = tmp_path / "voice_a.wav"
    sample_b = tmp_path / "voice_b.wav"
    _write_wav(sample_a, seconds=6.0, freq=220.0)
    _write_wav(sample_b, seconds=6.0, freq=330.0)
    prepared_a = prepare_voice_sample(sample_a, {"assets": tmp_path / "a" / "assets"})
    prepared_b = prepare_voice_sample(sample_b, {"assets": tmp_path / "b" / "assets"})
    scene = Scene(
        scene_id="ch_001_sc_001",
        chapter_id="ch_001",
        title="測試",
        source_pages=[1],
        narration_text_zh_tw="45 天前繳清 NT$3,000。",
        visual_type=VisualType.sketchbook_card,
        estimated_duration_sec=8.0,
        input_hash="hash",
        status=SceneStatus.pending,
    )
    cfg_a = {"provider": "xtts", "sample_rate_hz": 24000, "speaker_wav": prepared_a.processed_path}
    cfg_b = {"provider": "xtts", "sample_rate_hz": 24000, "speaker_wav": prepared_b.processed_path}
    assert _audio_input_hash(scene, cfg_a) != _audio_input_hash(scene, cfg_b)
