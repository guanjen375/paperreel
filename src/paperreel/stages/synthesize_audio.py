"""Stage 5 — TTS for each scene (resumable, per-scene retries)."""
from __future__ import annotations

from pathlib import Path

from ..hashing import hash_inputs
from ..io_utils import atomic_write_json, read_json
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


def _scene_done(scene: Scene) -> bool:
    return (
        scene.status in (SceneStatus.audio_done, SceneStatus.visual_done,
                         SceneStatus.rendered)
        and scene.audio_path is not None
        and Path(scene.audio_path).exists()
    )


def run(*, project_root: str | Path, db: StateDB, config: dict,
        resume: bool = True, max_retries: int = 2) -> SceneGraph:
    p = paths_for(project_root)
    graph = SceneGraph.model_validate(read_json(p["scene_graph"]))
    tts_cfg = config.get("tts", {})
    provider = make_tts_provider(tts_cfg)

    input_hash = hash_inputs("audio_v1",
                             tts_cfg.get("provider"),
                             tts_cfg.get("voice"),
                             tts_cfg.get("sample_rate_hz"),
                             tts_cfg.get("speaking_rate"))
    db.start_stage("audio", input_hash)

    new_scenes: list[Scene] = []
    failures: list[str] = []

    for sc in graph.scenes:
        out_path = p["audio_dir"] / f"{sc.scene_id}.wav"
        # Resume short-circuit
        if resume and sc.audio_path and Path(sc.audio_path).exists():
            new_scenes.append(sc)
            continue
        attempt = 0
        last_err: str | None = None
        actual: float | None = None
        while attempt <= max_retries:
            try:
                actual = provider.synthesize(
                    sc.narration_text_zh_tw, out_path,
                    voice=tts_cfg.get("voice"),
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

        sc = sc.model_copy(update={
            "audio_path": str(out_path),
            "actual_duration_sec": actual,
            "status": SceneStatus.audio_done,
            "last_error": None,
        })
        db.register_artifact(out_path, stage="audio", scene_id=sc.scene_id,
                             media_type="audio/wav", duration_sec=actual,
                             provenance={"voice": tts_cfg.get("voice"),
                                         "provider": tts_cfg.get("provider")})
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
