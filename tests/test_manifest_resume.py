"""Resume / cache invalidation driven by per-artifact manifest sidecars.

Old behaviour: ``Path.exists() → skip``. That silently served stale
output when the user changed the TTS voice, swapped speaker_wav, or
edited a scene's narration. The new behaviour writes an input_hash
into a ``<artifact>.manifest.json`` sidecar and only skips when that
hash matches what the current config would produce.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperreel.io_utils import read_json
from paperreel.manifest import (manifest_matches, manifest_path_for,
                                  read_manifest, write_manifest)
from paperreel.stages import (build_outline, build_scene_graph, ingest_pdf,
                               render_segments, render_visuals,
                               synthesize_audio, write_script)
from paperreel.state import StateDB


# --- manifest helpers ------------------------------------------------------

def test_manifest_path_appended_to_artifact(tmp_path: Path) -> None:
    """`.manifest.json` is appended, not substituted — so two artefacts
    that differ only in extension (e.g. wav vs m4a) keep distinct
    manifests, and the manifest can't accidentally shadow another file."""
    art = tmp_path / "foo.wav"
    assert manifest_path_for(art) == tmp_path / "foo.wav.manifest.json"


def test_manifest_round_trip(tmp_path: Path) -> None:
    art = tmp_path / "foo.wav"
    art.write_bytes(b"\x00")
    write_manifest(art, stage="audio", scene_id="ch_001_sc_001",
                   input_hash="abc123",
                   inputs={"narration": "hi", "speaker": "Ana"})
    m = read_manifest(art)
    assert m is not None
    assert m["input_hash"] == "abc123"
    assert m["scene_id"] == "ch_001_sc_001"
    assert m["inputs"]["speaker"] == "Ana"


def test_manifest_matches_strict(tmp_path: Path) -> None:
    art = tmp_path / "foo.wav"
    art.write_bytes(b"\x00")
    write_manifest(art, stage="audio", scene_id="s",
                   input_hash="abc", inputs={})
    assert manifest_matches(art, "abc")
    assert not manifest_matches(art, "different")


def test_manifest_matches_returns_false_when_artifact_missing(tmp_path: Path) -> None:
    art = tmp_path / "foo.wav"
    # manifest only, artefact gone (e.g. user deleted the wav by hand)
    manifest_path_for(art).write_text(
        json.dumps({"input_hash": "abc"}), encoding="utf-8"
    )
    assert not manifest_matches(art, "abc")


def test_manifest_matches_returns_false_when_manifest_corrupt(tmp_path: Path) -> None:
    art = tmp_path / "foo.wav"
    art.write_bytes(b"\x00")
    manifest_path_for(art).write_text("{ not json", encoding="utf-8")
    assert not manifest_matches(art, "anything")


# --- end-to-end helpers ----------------------------------------------------

def _drive_through_visuals(project_dir: Path, pdf: Path, cfg: dict):
    """Run ingest → outline → script → scenes → audio → visuals."""
    db = StateDB(project_dir / "state.sqlite")
    ingest_pdf.run(pdf_path=pdf, project_root=project_dir, db=db, config=cfg)
    build_outline.run(project_root=project_dir, project_name="t", db=db,
                      config=cfg, target_minutes="auto")
    write_script.run(project_root=project_dir, db=db, config=cfg)
    build_scene_graph.run(project_root=project_dir, project_name="t",
                          pdf_name=pdf.name, db=db, config=cfg)
    synthesize_audio.run(project_root=project_dir, db=db, config=cfg)
    render_visuals.run(project_root=project_dir, db=db, config=cfg)
    db.close()


# --- audio stage -----------------------------------------------------------

def test_audio_writes_manifest_with_voice_inputs(
    project_dir: Path, tiny_pdf: Path, test_cfg: dict
) -> None:
    db = StateDB(project_dir / "state.sqlite")
    ingest_pdf.run(pdf_path=tiny_pdf, project_root=project_dir, db=db, config=test_cfg)
    build_outline.run(project_root=project_dir, project_name="t", db=db,
                      config=test_cfg, target_minutes="auto")
    write_script.run(project_root=project_dir, db=db, config=test_cfg)
    build_scene_graph.run(project_root=project_dir, project_name="t",
                          pdf_name=tiny_pdf.name, db=db, config=test_cfg)
    g = synthesize_audio.run(project_root=project_dir, db=db, config=test_cfg)

    assert g.scenes, "fixture should produce at least one scene"
    for sc in g.scenes:
        if not sc.audio_path:
            continue
        m = read_manifest(sc.audio_path)
        assert m is not None, f"no manifest for {sc.audio_path}"
        assert m["stage"] == "audio"
        assert m["scene_id"] == sc.scene_id
        assert m["input_hash"]
        # The actual reason this PR exists: the speaker has to be part of
        # the recorded inputs so cache invalidation can detect a swap.
        assert "speaker" in m["inputs"]
        assert m["inputs"]["narration_sha256"]
    db.close()


def test_audio_rerenders_when_speaker_changes(
    project_dir: Path, tiny_pdf: Path, test_cfg: dict
) -> None:
    """Old logic: `Path.exists() → skip`, missed voice swaps. New logic
    treats speaker as part of the artefact's input fingerprint."""
    db = StateDB(project_dir / "state.sqlite")
    ingest_pdf.run(pdf_path=tiny_pdf, project_root=project_dir, db=db, config=test_cfg)
    build_outline.run(project_root=project_dir, project_name="t", db=db,
                      config=test_cfg, target_minutes="auto")
    write_script.run(project_root=project_dir, db=db, config=test_cfg)
    build_scene_graph.run(project_root=project_dir, project_name="t",
                          pdf_name=tiny_pdf.name, db=db, config=test_cfg)
    g = synthesize_audio.run(project_root=project_dir, db=db, config=test_cfg)

    audio_path = next(s.audio_path for s in g.scenes if s.audio_path)
    mtime_before = Path(audio_path).stat().st_mtime_ns
    hash_before = read_manifest(audio_path)["input_hash"]

    # Switch the speaker. With the old logic this was effectively a no-op
    # because the wav was still on disk.
    new_cfg = {**test_cfg, "tts": {**test_cfg["tts"], "speaker": "Different Voice"}}

    synthesize_audio.run(project_root=project_dir, db=db, config=new_cfg)

    mtime_after = Path(audio_path).stat().st_mtime_ns
    hash_after = read_manifest(audio_path)["input_hash"]
    assert mtime_after > mtime_before, "voice swap should have rebuilt the WAV"
    assert hash_after != hash_before, "manifest hash should reflect new speaker"
    assert read_manifest(audio_path)["inputs"]["speaker"] == "Different Voice"
    db.close()


def test_audio_skips_when_inputs_unchanged(
    project_dir: Path, tiny_pdf: Path, test_cfg: dict
) -> None:
    """Idempotent resume: same config + same scene graph → no rebuild."""
    db = StateDB(project_dir / "state.sqlite")
    ingest_pdf.run(pdf_path=tiny_pdf, project_root=project_dir, db=db, config=test_cfg)
    build_outline.run(project_root=project_dir, project_name="t", db=db,
                      config=test_cfg, target_minutes="auto")
    write_script.run(project_root=project_dir, db=db, config=test_cfg)
    build_scene_graph.run(project_root=project_dir, project_name="t",
                          pdf_name=tiny_pdf.name, db=db, config=test_cfg)
    g1 = synthesize_audio.run(project_root=project_dir, db=db, config=test_cfg)
    mtimes = {s.audio_path: Path(s.audio_path).stat().st_mtime_ns
              for s in g1.scenes if s.audio_path}

    g2 = synthesize_audio.run(project_root=project_dir, db=db, config=test_cfg)
    for s in g2.scenes:
        if s.audio_path and s.audio_path in mtimes:
            assert Path(s.audio_path).stat().st_mtime_ns == mtimes[s.audio_path]
    db.close()


def test_audio_rerenders_when_narration_changes(
    project_dir: Path, tiny_pdf: Path, test_cfg: dict
) -> None:
    """The narration_sha256 is part of the manifest hash, so editing
    script.json + rebuilding the scene graph must invalidate audio."""
    db = StateDB(project_dir / "state.sqlite")
    ingest_pdf.run(pdf_path=tiny_pdf, project_root=project_dir, db=db, config=test_cfg)
    build_outline.run(project_root=project_dir, project_name="t", db=db,
                      config=test_cfg, target_minutes="auto")
    write_script.run(project_root=project_dir, db=db, config=test_cfg)
    build_scene_graph.run(project_root=project_dir, project_name="t",
                          pdf_name=tiny_pdf.name, db=db, config=test_cfg)
    g = synthesize_audio.run(project_root=project_dir, db=db, config=test_cfg)
    audio_path = next(s.audio_path for s in g.scenes if s.audio_path)
    mtime_before = Path(audio_path).stat().st_mtime_ns

    # Edit narration directly on the scene graph and rerun audio.
    graph_path = project_dir / "intermediate" / "scene_graph.json"
    data = read_json(graph_path)
    target_id = next(s["scene_id"] for s in data["scenes"]
                     if s["audio_path"] == audio_path)
    for s in data["scenes"]:
        if s["scene_id"] == target_id:
            s["narration_text_zh_tw"] = "整段旁白已被覆寫，這應該觸發重新合成。"
            # Force the stage to consider re-rendering — otherwise it
            # would short-circuit on the audio_path it already knows.
            s["audio_path"] = None
            break
    graph_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    synthesize_audio.run(project_root=project_dir, db=db, config=test_cfg)
    assert Path(audio_path).stat().st_mtime_ns > mtime_before
    db.close()


# --- visual stage ----------------------------------------------------------

def test_visual_writes_manifest(
    project_dir: Path, tiny_pdf: Path, test_cfg: dict
) -> None:
    _drive_through_visuals(project_dir, tiny_pdf, test_cfg)
    visuals = list((project_dir / "assets" / "visuals").glob("*.png"))
    assert visuals
    for png in visuals:
        m = read_manifest(png)
        assert m is not None, f"no manifest for {png}"
        assert m["stage"] == "visuals"
        assert m["input_hash"]
        assert m["inputs"]["renderer"]["resolution"]
        assert m["inputs"]["schema"] == "visual_artifact_v5"
        assert m["inputs"]["layout_version"]


def test_visual_manifest_records_layout_version(
    project_dir: Path, tiny_pdf: Path, test_cfg: dict
) -> None:
    _drive_through_visuals(project_dir, tiny_pdf, test_cfg)
    png = next(iter((project_dir / "assets" / "visuals").glob("*.png")))
    m = read_manifest(png)
    assert m["inputs"]["schema"] == "visual_artifact_v5"
    assert m["inputs"]["layout_version"] == "card_layout_v1"


def test_visual_rerenders_when_renderer_color_changes(
    project_dir: Path, tiny_pdf: Path, test_cfg: dict
) -> None:
    _drive_through_visuals(project_dir, tiny_pdf, test_cfg)
    png = next(iter((project_dir / "assets" / "visuals").glob("*.png")))
    before_mtime = png.stat().st_mtime_ns
    before_hash = read_manifest(png)["input_hash"]

    new_cfg = {**test_cfg, "renderer": {**test_cfg["renderer"],
                                         "background_color": "#FF0000"}}
    db = StateDB(project_dir / "state.sqlite")
    render_visuals.run(project_root=project_dir, db=db, config=new_cfg)
    db.close()

    after_mtime = png.stat().st_mtime_ns
    after_hash = read_manifest(png)["input_hash"]
    assert after_mtime > before_mtime
    assert after_hash != before_hash


def test_visual_skip_when_unchanged(
    project_dir: Path, tiny_pdf: Path, test_cfg: dict
) -> None:
    _drive_through_visuals(project_dir, tiny_pdf, test_cfg)
    pngs = list((project_dir / "assets" / "visuals").glob("*.png"))
    mtimes = {p: p.stat().st_mtime_ns for p in pngs}

    db = StateDB(project_dir / "state.sqlite")
    render_visuals.run(project_root=project_dir, db=db, config=test_cfg)
    db.close()

    for p, mt in mtimes.items():
        assert p.stat().st_mtime_ns == mt, f"{p} re-rendered without input change"


# --- segment stage uses upstream manifests ---------------------------------

def test_segment_input_hash_composes_audio_hash() -> None:
    """The segment hash must include the upstream audio's input_hash so
    when the audio rebuilds, the segment automatically rebuilds too."""
    from paperreel.models import Scene, VisualType
    from paperreel.stages.render_segments import (
        _segment_codec_signature, _segment_inputs, _segment_input_hash)
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tdir:
        audio_path = Path(tdir) / "a.wav"
        visual_path = Path(tdir) / "v.png"
        audio_path.write_bytes(b"audio")
        visual_path.write_bytes(b"visual")

        write_manifest(audio_path, stage="audio", scene_id="s",
                       input_hash="audio_hash_v1", inputs={})
        write_manifest(visual_path, stage="visuals", scene_id="s",
                       input_hash="visual_hash_v1", inputs={})

        sc = Scene(
            scene_id="s", chapter_id="c", title="t", source_pages=[1],
            narration_text_zh_tw="x",
            visual_type=VisualType.bullet_card,
            estimated_duration_sec=30.0,
            actual_duration_sec=25.0,
            audio_path=str(audio_path),
            visual_asset_paths=[str(visual_path)],
            input_hash="scene_input_hash",
        )
        codec_sig = _segment_codec_signature(
            {"video_codec": "libx264", "audio_codec": "aac",
             "pixel_format": "yuv420p"}, fps=30, res=(1920, 1080),
        )
        inputs = _segment_inputs(sc, codec_sig)
        assert inputs["audio_input_hash"] == "audio_hash_v1"
        assert inputs["visual_input_hash"] == "visual_hash_v1"
        h1 = _segment_input_hash(sc, codec_sig)

        # Bump the audio manifest, hash should change.
        write_manifest(audio_path, stage="audio", scene_id="s",
                       input_hash="audio_hash_v2", inputs={})
        h2 = _segment_input_hash(sc, codec_sig)
        assert h1 != h2, "segment hash must move when upstream audio's hash moves"
