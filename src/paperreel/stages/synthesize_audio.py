"""Stage 5 — TTS for each scene (resumable, per-scene retries).

Resume model: each ``assets/audio/<scene_id>.wav`` has a sidecar
``.manifest.json`` carrying the input_hash that produced it (narration,
speaker, speaker_wav SHA, language, sample rate, speaking rate, provider).
If the hash matches what the current config would produce, we skip;
otherwise we resynthesize. "File exists" alone is no longer sufficient
because the previous logic missed voice / speaker_wav / language changes.
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

from ..hashing import hash_inputs, sha256_text
from ..io_utils import atomic_write_json, read_json
from ..manifest import manifest_matches, sha256_of, write_manifest
from ..models import Scene, SceneGraph, SceneStatus
from ..providers.tts_base import make_tts_provider
from ..state import StateDB


def paths_for(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    return {
        "scene_graph": root / "intermediate" / "scene_graph.json",
        "audio_dir": root / "assets" / "audio",
        "scene_graph_out": root / "intermediate" / "scene_graph.json",   # rewritten in-place
    }


def _stage_config_hash(tts_cfg: dict) -> str:
    """Coarse hash used as the stage-level input_hash in the DB. Per-scene
    cache invalidation uses :func:`_audio_inputs` below, which is what
    actually decides skip/rebuild."""
    return hash_inputs(
        "audio_stage_v2",
        tts_cfg.get("provider"),
        tts_cfg.get("speaker"),
        tts_cfg.get("speaker_wav"),
        sha256_of(tts_cfg.get("speaker_wav")),
        tts_cfg.get("language"),
        tts_cfg.get("sample_rate_hz"),
        tts_cfg.get("speaking_rate"),
    )


def _audio_inputs(scene: Scene, tts_cfg: dict) -> dict[str, Any]:
    """Return the inputs dict that fully fingerprints one audio artifact.

    Anything that can change the produced waveform belongs here. The
    SHA of ``speaker_wav`` matters because users sometimes edit / replace
    the reference clip without renaming it — the previous resume logic
    couldn't catch that.
    """
    return {
        "schema": "audio_artifact_v3",
        "text_chunking": "xtts_zh_safe_80",
        "narration_sha256": sha256_text(scene.narration_text_zh_tw),
        "narration_len": len(scene.narration_text_zh_tw),
        "language": tts_cfg.get("language"),
        "speaker": tts_cfg.get("speaker"),
        "speaker_wav": tts_cfg.get("speaker_wav"),
        "speaker_wav_sha256": sha256_of(tts_cfg.get("speaker_wav")),
        "sample_rate_hz": int(tts_cfg.get("sample_rate_hz", 48000)),
        "speaking_rate": float(tts_cfg.get("speaking_rate", 1.0)),
        "provider": tts_cfg.get("provider"),
    }


def _audio_input_hash(scene: Scene, tts_cfg: dict) -> str:
    return hash_inputs("audio_artifact_v3", _audio_inputs(scene, tts_cfg))


def _wav_duration_seconds(path: Path) -> float | None:
    """Cheap WAV duration probe for resume — avoids dragging ffprobe in
    for cached scenes. Returns None on parse failure (caller falls
    back to the scene's prior recorded duration)."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / max(1, w.getframerate())
    except Exception:
        return None


def run(*, project_root: str | Path, db: StateDB, config: dict,
        resume: bool = True, max_retries: int = 2) -> SceneGraph:
    p = paths_for(project_root)
    graph = SceneGraph.model_validate(read_json(p["scene_graph"]))
    tts_cfg = config.get("tts", {})
    provider = make_tts_provider(tts_cfg)

    db.start_stage("audio", _stage_config_hash(tts_cfg))

    new_scenes: list[Scene] = []
    failures: list[str] = []

    for sc in graph.scenes:
        out_path = p["audio_dir"] / f"{sc.scene_id}.wav"
        expected_hash = _audio_input_hash(sc, tts_cfg)

        # Resume: only skip if the existing file's manifest still matches
        # what we'd produce with the current config + narration. A bare
        # "file exists" check would also serve stale audio when the user
        # swapped speakers / edited narration / replaced speaker_wav.
        if resume and manifest_matches(out_path, expected_hash):
            duration = sc.actual_duration_sec or _wav_duration_seconds(out_path)
            sc = sc.model_copy(update={
                "audio_path": str(out_path),
                "actual_duration_sec": duration,
                "status": (SceneStatus.audio_done
                           if sc.status == SceneStatus.pending else sc.status),
            })
            new_scenes.append(sc)
            continue

        attempt = 0
        last_err: str | None = None
        actual: float | None = None
        while attempt <= max_retries:
            try:
                actual = provider.synthesize(
                    sc.narration_text_zh_tw, out_path,
                    voice=tts_cfg.get("speaker") or tts_cfg.get("voice"),
                    sample_rate_hz=int(tts_cfg.get("sample_rate_hz", 48000)),
                    speaking_rate=float(tts_cfg.get("speaking_rate", 1.0)),
                )
                break
            except Exception as e:
                attempt += 1
                last_err = repr(e)
                if attempt > max_retries:
                    break
        if actual is None:
            sc = sc.model_copy(update={
                "status": SceneStatus.failed,
                "last_error": last_err,
                "retry_count": sc.retry_count + attempt,
            })
            db.upsert_scene(sc.scene_id, sc.chapter_id, sc.status.value,
                            sc.input_hash, sc.model_dump(mode="json"),
                            last_error=last_err, bump_retry=True)
            db.log_error("audio", last_err or "tts failed", scene_id=sc.scene_id)
            failures.append(sc.scene_id)
            new_scenes.append(sc)
            continue

        write_manifest(
            out_path,
            stage="audio",
            scene_id=sc.scene_id,
            input_hash=expected_hash,
            inputs=_audio_inputs(sc, tts_cfg),
            extra={"actual_duration_sec": actual},
        )

        sc = sc.model_copy(update={
            "audio_path": str(out_path),
            "actual_duration_sec": actual,
            "status": SceneStatus.audio_done,
            "last_error": None,
        })
        db.register_artifact(out_path, stage="audio", scene_id=sc.scene_id,
                             media_type="audio/wav", duration_sec=actual,
                             provenance={"speaker": tts_cfg.get("speaker"),
                                         "speaker_wav": tts_cfg.get("speaker_wav"),
                                         "provider": tts_cfg.get("provider"),
                                         "input_hash": expected_hash})
        db.upsert_scene(sc.scene_id, sc.chapter_id, sc.status.value,
                        sc.input_hash, sc.model_dump(mode="json"))
        new_scenes.append(sc)

    graph = graph.model_copy(update={"scenes": new_scenes})
    atomic_write_json(p["scene_graph_out"], graph.model_dump(mode="json"))
    db.finish_stage("audio", [str(p["scene_graph_out"])])
    if failures:
        # Stage itself does not fail — individual scenes are flagged.
        db.log_error("audio", f"audio failed for scenes: {failures}")
    return graph
