from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

import paperreel.cli as cli
from _fakes.tts_fake import FakeTTS


HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _write_wav(path: Path, *, seconds: float = 6.0, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        for i in range(int(seconds * rate)):
            sample = int(6000 * math.sin(2 * math.pi * 220.0 * i / rate))
            w.writeframes(struct.pack("<h", sample))


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_voice_test_calls_fake_tts_with_processed_speaker_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = tmp_path / "my_voice.wav"
    out = tmp_path / "voice_test.wav"
    _write_wav(sample)
    seen: dict[str, str] = {}

    class CapturingFakeTTS(FakeTTS):
        def __init__(self, cfg: dict | None = None):
            super().__init__(cfg)
            seen["speaker_wav"] = str((cfg or {}).get("speaker_wav"))
            seen["normalize_zh_tw"] = str((cfg or {}).get("normalize_zh_tw"))

    monkeypatch.setattr(cli, "make_tts_provider", lambda cfg: CapturingFakeTTS(cfg))
    result = CliRunner().invoke(
        cli._typer_app,
        ["voice-test", "--voice-sample", str(sample), "--out", str(out)],
    )

    assert result.exit_code == 0, result.stdout
    assert out.exists()
    assert seen["speaker_wav"].endswith(".speaker.wav")
    assert Path(seen["speaker_wav"]).exists()
    assert seen["normalize_zh_tw"] == "True"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_cli_voice_sample_overrides_config_voice_sample(tmp_path: Path) -> None:
    config_sample = tmp_path / "config_voice.wav"
    cli_sample = tmp_path / "cli_voice.wav"
    _write_wav(config_sample, seconds=6.0)
    _write_wav(cli_sample, seconds=6.0)
    cfg = {"tts": {"voice_sample": str(config_sample)}}

    prepared = cli._apply_voice_sample(
        cfg, {"assets": tmp_path / "project" / "assets"}, str(cli_sample)
    )

    assert prepared is not None
    assert prepared.original_path == str(cli_sample.resolve())
    assert cfg["tts"]["voice_sample"] == str(cli_sample)
    assert cfg["tts"]["speaker_wav"] == prepared.processed_path


def test_default_voice_info_line_when_no_voice_sample() -> None:
    line = cli._voice_source_info_line({"tts": {"speaker": "Ana Florence"}}, None)
    assert line == "[INFO] 未提供 voice_sample，使用 XTTS 預設聲音: Ana Florence"


def test_advanced_speaker_wav_info_line_when_configured() -> None:
    line = cli._voice_source_info_line({"tts": {"speaker_wav": "/tmp/ref.wav"}}, None)
    assert line == "[INFO] 使用進階 speaker_wav: /tmp/ref.wav"
